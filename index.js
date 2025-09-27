// =================================================================================
// IMPORTS AND CLIENT INITIALIZATION
// =================================================================================
const { SecretManagerServiceClient } = require('@google-cloud/secret-manager');
const { Storage } = require('@google-cloud/storage');
const { ImageAnnotatorClient } = require('@google-cloud/vision');
const { VertexAI } = require('@google-cloud/vertexai');
const axios = require('axios');
const path = require('path');
const crypto = require('crypto');

const storage = new Storage();
const secretManager = new SecretManagerServiceClient();
const visionClient = new ImageAnnotatorClient();

// Initialize Vertex AI with your specific Project ID
const location = 'us-central1'; // Or your preferred GCP location
const vertexAI = new VertexAI({
  project: 'your-gcp-project-id-here', // 👈 **REPLACE THIS with your actual Project ID**
  location: location,
});
const generativeModel = vertexAI.getGenerativeModel({ model: 'gemini-1.5-flash-001' });

// =================================================================================
// GLOBAL CONFIGURATION
// =================================================================================

const WF_CONFIG = {
    // Buckets
    COMPLETED_BUCKET_NAME: 'files-completed',
    ERROR_BUCKET_NAME: 'files-error',

    // Qdrant Collections
    QDRANT_CHUNK_COLLECTION: 'nexus_shards',
    QDRANT_DOC_COLLECTION: 'cortex_nodes',

    // AI Models
    EMBEDDING_MODEL_OPENAI: 'text-embedding-3-small',
    EMBEDDING_MODEL_GEMINI: 'embedding-001',
    SUMMARY_MODEL: 'gpt-4o-mini',

    // Processing Parameters
    CHUNK_SIZE: 500, // In words
    CHUNK_OVERLAP: 50, // In words
    SUMMARY_TEMPERATURE: 0.2,
    SUMMARY_MAX_TOKENS: 1400,

    // Normalization
    NORMALIZER_VERSION: "3.2.0",
};

const ORGANIZE_CONFIG = {
    ORGANIZED_BUCKET: 'files-organized-for-indexing',
    PHOTO_BUCKET: 'files-photos',
    SKIPPED_BUCKET: 'files-skipped',
    DUPLICATE_BUCKET: 'files-duplicates',
    FILE_RENAME_PROMPT: `You are an expert file naming assistant. Based on the following text, propose a single, concise, professional filename **base** (without the file extension).
- The filename should clearly reflect the document's main subject and date.
- Use underscores instead of spaces.
- Format dates as YYYY-MM-DD.
- Append "_v01" to the end of the base name.
- Output ONLY the base filename.
Example Output: 372_Fifth_Avenue_Owners_Inc_Financial_Statements_2023-2024_v01`
};

// =================================================================================
// 📁 FUNCTION 1: organizeFile (Deduplicator, Renamer & Sorter)
// =================================================================================

exports.organizeFile = async (file, context) => {
    const sourceBucketName = file.bucket;
    const sourceFileName = file.name;
    const contentType = file.contentType;
    const gcpProject = process.env.GCP_PROJECT;

    console.log(`[Organizer] 🚀 Received file: ${sourceFileName}`);
    const sourceFile = storage.bucket(sourceBucketName).file(sourceFileName);

    try {
        // --- 1. Deduplication Check ---
        console.log(`[Organizer] 🔍 Computing hash for ${sourceFileName}...`);
        const fileHash = await computeFileHash(sourceBucketName, sourceFileName);
        const apiKeys = await getApiKeys(gcpProject); // Need keys for Qdrant check

        if (await checkIfHashExists(fileHash, apiKeys)) {
            console.log(`[Organizer] duplicate found (Hash: ${fileHash}). Moving to duplicates bucket.`);
            await sourceFile.move(storage.bucket(ORGANIZE_CONFIG.DUPLICATE_BUCKET).file(sourceFileName));
            return; // Stop processing
        }
        console.log(`[Organizer] ✅ Unique file detected (Hash: ${fileHash}).`);

        // --- 2. Sort Photos ---
        if (contentType && contentType.startsWith('image/')) {
            await sourceFile.move(storage.bucket(ORGANIZE_CONFIG.PHOTO_BUCKET).file(sourceFileName));
            console.log(`[Organizer] 📸 Moved photo to ${ORGANIZE_CONFIG.PHOTO_BUCKET}`);
            return;
        }

        // --- 3. Extract Text & Rename ---
        const supportedTextMimes = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
        if (!supportedTextMimes.includes(contentType)) {
            await sourceFile.move(storage.bucket(ORGANIZE_CONFIG.SKIPPED_BUCKET).file(sourceFileName));
            console.log(`[Organizer] ⏩ Moved unsupported file to ${ORGANIZE_CONFIG.SKIPPED_BUCKET}`);
            return;
        }

        let textContent = await getTextFromImageOrPdf(sourceBucketName, sourceFileName);
        if (!textContent || textContent.length < 20) {
            await sourceFile.move(storage.bucket(ORGANIZE_CONFIG.SKIPPED_BUCKET).file(sourceFileName));
            console.log(`[Organizer] ⚠️ Not enough content. Moved to skipped.`);
            return;
        }

        const result = await generativeModel.generateContent({
            contents: [{ role: 'user', parts: [{ text: ORGANIZE_CONFIG.FILE_RENAME_PROMPT }, { text: textContent.slice(0, 15000) }] }]
        });
        
        const suggestedBaseName = result.response.candidates[0].content.parts[0].text.trim().replace(/["']/g, '');
        if (!suggestedBaseName) throw new Error('AI did not suggest a name.');

        const newName = `${suggestedBaseName}${path.extname(sourceFileName)}`;

        // --- 4. Move to Organized Bucket ---
        await sourceFile.rename(storage.bucket(ORGANIZE_CONFIG.ORGANIZED_BUCKET).file(newName));
        console.log(`[Organizer] ✅ Renamed and Moved: "${sourceFileName}" -> "${newName}"`);

    } catch (error) {
        console.error(`[Organizer] ❌ Error processing ${sourceFileName}:`, error);
        await sourceFile.move(storage.bucket(ORGANIZE_CONFIG.SKIPPED_BUCKET).file(`ERROR_${sourceFileName}`));
    }
};

// =================================================================================
// 🗂️ FUNCTION 2: processFileToQdrant (The Indexer)
// =================================================================================

exports.processFileToQdrant = async (file, context) => {
    const sourceBucketName = file.bucket;
    const sourceFileName = file.name;
    const fileId = `${sourceBucketName}/${sourceFileName}`;
    const gcpProject = process.env.GCP_PROJECT;
    
    console.log(`[Indexer] 🚀 Starting processing for unique file: ${fileId}`);
    
    try {
        const apiKeys = await getApiKeys(gcpProject);
        
        // Ensure the necessary payload index exists in Qdrant before processing.
        await ensureQdrantIndexes(apiKeys);

        // Compute hash to add to payload
        const fileHash = await computeFileHash(sourceBucketName, sourceFileName);
        let textContent = await getTextFromImageOrPdf(sourceBucketName, sourceFileName);

        const chunks = chunkText(textContent);
        
        const [openaiEmbeddings, geminiEmbeddings, chunkSummaries] = await Promise.all([
            getOpenAIEmbeddings(chunks, apiKeys.openAIKey),
            getGeminiEmbeddings(chunks, apiKeys.geminiKey),
            getChunkSummaries(chunks, apiKeys.openAIKey)
        ]);
        const docSummary = await getDocSummary(chunkSummaries, apiKeys.openAIKey);

        const chunkPoints = createChunkPoints(fileId, sourceFileName, chunks, openaiEmbeddings, geminiEmbeddings, chunkSummaries, fileHash);
        const docPoint = createDocPoint(fileId, sourceFileName, chunks, openaiEmbeddings, geminiEmbeddings, docSummary, fileHash);
        
        await Promise.all([
            deletePointsByFileId(fileId, WF_CONFIG.QDRANT_CHUNK_COLLECTION, apiKeys),
            deletePointsByFileId(fileId, WF_CONFIG.QDRANT_DOC_COLLECTION, apiKeys)
        ]);

        await Promise.all([
            upsertPoints(chunkPoints, WF_CONFIG.QDRANT_CHUNK_COLLECTION, apiKeys),
            upsertPoints([docPoint], WF_CONFIG.QDRANT_DOC_COLLECTION, apiKeys)
        ]);
        
        await storage.bucket(sourceBucketName).file(sourceFileName).move(storage.bucket(WF_CONFIG.COMPLETED_BUCKET_NAME).file(sourceFileName));
        console.log(`[Indexer] ✅ Successfully processed and moved ${sourceFileName}.`);
    } catch (error) {
        console.error(`[Indexer] ❌ Failed to process file ${sourceFileName}:`, error.response ? error.response.data : error);
        await storage.bucket(sourceBucketName).file(sourceFileName).move(storage.bucket(WF_CONFIG.ERROR_BUCKET_NAME).file(sourceFileName));
    }
};

// =================================================================================
// HELPER FUNCTIONS
// =================================================================================

async function getApiKeys(projectId) {
    const secrets = ['OPENAI_API_KEY', 'GEMINI_API_KEY', 'QDRANT_API_KEY', 'QDRANT_API_URL'];
    const results = await Promise.all(
        secrets.map(secret => secretManager.accessSecretVersion({
            name: `projects/${projectId}/secrets/${secret}/versions/latest`
        }))
    );
    return {
        openAIKey: results[0][0].payload.data.toString(),
        geminiKey: results[1][0].payload.data.toString(),
        qdrantKey: results[2][0].payload.data.toString(),
        qdrantUrl: results[3][0].payload.data.toString(),
    };
}

function computeFileHash(bucketName, fileName) {
    return new Promise((resolve, reject) => {
        const hash = crypto.createHash('sha256');
        const stream = storage.bucket(bucketName).file(fileName).createReadStream();
        stream.on('data', (data) => hash.update(data));
        stream.on('end', () => resolve(hash.digest('hex')));
        stream.on('error', reject);
    });
}

async function checkIfHashExists(hash, { qdrantUrl, qdrantKey }) {
    const url = `${qdrantUrl}/collections/${WF_CONFIG.QDRANT_DOC_COLLECTION}/points/scroll`;
    const payload = {
        filter: { must: [{ key: "content_hash", match: { value: hash } }] },
        limit: 1,
        with_payload: false,
        with_vector: false
    };
    try {
        const response = await axios.post(url, payload, { headers: { 'api-key': qdrantKey } });
        return response.data.result.points.length > 0;
    } catch (error) {
        console.error("Error checking for hash in Qdrant:", error.response?.data || error.message);
        return false;
    }
}

async function ensureQdrantIndexes(apiKeys) {
    const { qdrantUrl, qdrantKey } = apiKeys;
    const headers = { 'api-key': qdrantKey, 'Content-Type': 'application/json' };
    const fieldName = "content_hash";
    const collectionName = WF_CONFIG.QDRANT_DOC_COLLECTION;
    const url = `${qdrantUrl}/collections/${collectionName}/index`;

    try {
        await axios.put(url, { field_name: fieldName, field_schema: "keyword" }, { headers });
        console.log(`[Indexer] ✅ Ensured payload index exists for 'content_hash' in ${collectionName}.`);
    } catch (error) {
        if (error.response?.status !== 409) { // 409 Conflict means it already exists
            console.error(`[Indexer] ⚠️ Could not create Qdrant payload index:`, error.response?.data || error.message);
        }
    }
}

async function getTextFromImageOrPdf(bucketName, fileName) {
    console.log(`[Helper] 👁️ Performing OCR on gs://${bucketName}/${fileName}`);
    try {
        const [result] = await visionClient.documentTextDetection(`gs://${bucketName}/${fileName}`);
        const detection = result.fullTextAnnotation;
        return detection ? detection.text : '';
    } catch (e) {
        console.error(`[Helper] OCR failed for ${fileName}: ${e.message}`);
        return '';
    }
}

function chunkText(text) {
    const words = text.split(/\s+/).filter(Boolean);
    const chunks = [];
    const step = WF_CONFIG.CHUNK_SIZE - WF_CONFIG.CHUNK_OVERLAP;
    for (let i = 0; i < words.length; i += step) {
        chunks.push(words.slice(i, i + WF_CONFIG.CHUNK_SIZE).join(" "));
    }
    return chunks;
}

async function getOpenAIEmbeddings(chunks, apiKey) {
    const response = await axios.post('https://api.openai.com/v1/embeddings', {
        input: chunks, model: WF_CONFIG.EMBEDDING_MODEL_OPENAI
    }, { headers: { 'Authorization': `Bearer ${apiKey}` } });
    return response.data.data.map(item => item.embedding);
}

async function getGeminiEmbeddings(chunks, apiKey) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${WF_CONFIG.EMBEDDING_MODEL_GEMINI}:batchEmbedText?key=${apiKey}`;
    const requests = chunks.map(chunk => ({ model: `models/${WF_CONFIG.EMBEDDING_MODEL_GEMINI}`, content: { parts: [{ text: chunk }] } }));
    const response = await axios.post(url, { requests });
    return response.data.embeddings.map(item => item.values);
}

async function getChunkSummaries(chunks, apiKey) {
    const promises = chunks.map(chunk => axios.post('https://api.openai.com/v1/chat/completions', {
        model: WF_CONFIG.SUMMARY_MODEL,
        messages: [{ role: 'system', content: 'You are a meticulous document analyst. Summarize the following text, capturing all key details, figures, and names precisely.' }, { role: 'user', content: chunk }],
        temperature: WF_CONFIG.SUMMARY_TEMPERATURE, max_tokens: WF_CONFIG.SUMMARY_MAX_TOKENS,
    }, { headers: { 'Authorization': `Bearer ${apiKey}` } }));
    
    const responses = await Promise.all(promises);
    return responses.map(response => response.data.choices[0].message.content.trim());
}

async function getDocSummary(chunkSummaries, apiKey) {
    const topSummaries = chunkSummaries.slice(0, 12).map((s, i) => `--- CHUNK ${i} ---\n${s}`).join("\n\n");
    const prompt = `Combine the following chunk summaries into a single, comprehensive document-level summary.\n\n${topSummaries}`;
    
    const response = await axios.post('https://api.openai.com/v1/chat/completions', {
        model: WF_CONFIG.SUMMARY_MODEL,
        messages: [{ role: 'system', content: 'You are a senior analyst. Write a comprehensive, well-structured document-level summary.' }, { role: 'user', content: prompt }],
        temperature: WF_CONFIG.SUMMARY_TEMPERATURE, max_tokens: WF_CONFIG.SUMMARY_MAX_TOKENS
    }, { headers: { 'Authorization': `Bearer ${apiKey}` } });
    return response.data.choices[0].message.content.trim();
}

async function upsertPoints(points, collectionName, { qdrantUrl, qdrantKey }) {
    if (!points || points.length === 0) return;
    const url = `${qdrantUrl}/collections/${collectionName}/points?wait=true`;
    await axios.put(url, { points }, { headers: { 'api-key': qdrantKey } });
}

async function deletePointsByFileId(fileId, collectionName, { qdrantUrl, qdrantKey }) {
    const url = `${qdrantUrl}/collections/${collectionName}/points/delete?wait=true`;
    const payload = { filter: { must: [{ key: "file_id", match: { value: fileId } }] } };
    await axios.post(url, payload, { headers: { 'api-key': qdrantKey } });
}

function createChunkPoints(fileId, fileName, chunks, openaiEmbeddings, geminiEmbeddings, summaries, fileHash) {
    return chunks.map((chunk, i) => {
        const payload = {
            file_id: fileId,
            file_name: fileName,
            chunk_text: chunk,
            ai_summary: summaries[i],
            chunk_index: i,
            total_chunks: chunks.length,
            processed_at: new Date().toISOString(),
            content_hash: fileHash
        };
        return {
            id: crypto.randomUUID(),
            vector: { openai: openaiEmbeddings[i], gemini: geminiEmbeddings[i] },
            payload
        };
    });
}

function createDocPoint(fileId, fileName, chunks, openaiEmbeddings, geminiEmbeddings, docSummary, fileHash) {
    const meanVec = (arr) => {
        if (!arr || !arr.length) return [];
        const dim = arr[0].length;
        const sum = new Array(dim).fill(0);
        for (const vec of arr) { for (let i = 0; i < dim; i++) sum[i] += vec[i]; }
        return sum.map(v => v / arr.length);
    };
    const payload = {
        file_id: fileId,
        file_name: fileName,
        doc_summary: docSummary,
        chunk_count: chunks.length,
        processed_at: new Date().toISOString(),
        content_hash: fileHash
    };
    return {
        id: crypto.randomUUID(),
        vector: { openai: meanVec(openaiEmbeddings), gemini: meanVec(geminiEmbeddings) },
        payload
    };
}
