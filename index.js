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

// Initialize Vertex AI
const project = process.env.GCP_PROJECT;
const location = 'us-central1';
const vertexAI = new VertexAI({ project, location });
const generativeModel = vertexAI.getGenerativeModel({ model: 'gemini-1.5-flash-001' });

// =================================================================================
// 📁 FUNCTION 1: organizeFile (Deduplicator, Renamer & Sorter)
// =================================================================================

const ORGANIZE_CONFIG = {
    ORGANIZED_BUCKET: 'files-organized-for-indexing',
    PHOTO_BUCKET: 'files-photos',
    SKIPPED_BUCKET: 'files-skipped',
    DUPLICATE_BUCKET: 'files-duplicates', // New bucket for duplicates
    FILE_RENAME_PROMPT: `You are an expert file naming assistant...` // (Prompt from previous step)
};

exports.organizeFile = async (file, context) => {
    const sourceBucketName = file.bucket;
    const sourceFileName = file.name;
    const contentType = file.contentType;

    console.log(`[Organizer] 🚀 Received file: ${sourceFileName}`);
    const sourceFile = storage.bucket(sourceBucketName).file(sourceFileName);

    try {
        // --- 1. Deduplication Check (NEW) ---
        console.log(`[Organizer] 🔍 Computing hash for ${sourceFileName}...`);
        const fileHash = await computeFileHash(sourceBucketName, sourceFileName);
        const apiKeys = await getApiKeys(project); // Need keys for Qdrant check

        if (await checkIfHashExists(fileHash, apiKeys)) {
            console.log(`[Organizer]  Duplicate found (Hash: ${fileHash}). Moving to duplicates bucket.`);
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
        // (This section remains the same as before)
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
        
        const suggestedBaseName = result.response.candidates[0].content.parts[0].text.trim();
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
    // ... (This function's main logic remains largely the same) ...
    const sourceBucketName = file.bucket;
    const sourceFileName = file.name;
    const fileId = `${sourceBucketName}/${sourceFileName}`;
    console.log(`[Indexer] 🚀 Starting processing for unique file: ${fileId}`);
    
    try {
        const apiKeys = await getApiKeys(project);
        
        // Ensure the necessary payload index exists in Qdrant before processing.
        await ensureQdrantIndexes(apiKeys);

        // --- ADD HASH TO PAYLOAD (NEW) ---
        const fileHash = await computeFileHash(sourceBucketName, sourceFileName);
        let textContent = await getTextFromImageOrPdf(sourceBucketName, sourceFileName);

        // ... (rest of the function: chunking, embeddings, summaries)
        const chunks = chunkText(textContent);
        const [openaiEmbeddings, geminiEmbeddings, chunkSummaries] = await Promise.all([/*...*/]);
        const docSummary = await getDocSummary(chunkSummaries, apiKeys.openAIKey);

        // Pass the hash to the point creation functions
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
        
        await storage.bucket(sourceBucketName).file(sourceFileName).move(/* to completed bucket */);
        console.log(`[Indexer] ✅ Successfully processed and moved ${sourceFileName}.`);
    } catch (error) {
        console.error(`[Indexer] ❌ Failed to process file ${sourceFileName}:`, error);
        await storage.bucket(sourceBucketName).file(sourceFileName).move(/* to error bucket */);
    }
};

// =================================================================================
// HELPER FUNCTIONS (New and Modified)
// =================================================================================

/**
 * Computes the SHA-256 hash of a file in GCS.
 */
function computeFileHash(bucketName, fileName) {
    return new Promise((resolve, reject) => {
        const hash = crypto.createHash('sha256');
        const stream = storage.bucket(bucketName).file(fileName).createReadStream();
        stream.on('data', (data) => hash.update(data));
        stream.on('end', () => resolve(hash.digest('hex')));
        stream.on('error', reject);
    });
}

/**
 * Checks Qdrant to see if a point with a given hash already exists.
 */
async function checkIfHashExists(hash, { qdrantUrl, qdrantKey }) {
    const url = `${qdrantUrl}/collections/${WF_CONFIG.QDRANT_DOC_COLLECTION}/points/scroll`;
    const payload = {
        filter: {
            must: [{ key: "content_hash", match: { value: hash } }]
        },
        limit: 1,
        with_payload: false,
        with_vector: false
    };
    try {
        const response = await axios.post(url, payload, {
            headers: { 'api-key': qdrantKey }
        });
        return response.data.result.points.length > 0;
    } catch (error) {
        console.error("Error checking for hash in Qdrant:", error.response?.data || error.message);
        // Fail-safe: assume it doesn't exist to avoid blocking new files if Qdrant is down.
        return false;
    }
}

/**
 * Ensures the required payload indexes exist in Qdrant collections.
 */
async function ensureQdrantIndexes(apiKeys) {
    const { qdrantUrl, qdrantKey } = apiKeys;
    const headers = { 'api-key': qdrantKey, 'Content-Type': 'application/json' };
    const fieldName = "content_hash";
    const collectionName = WF_CONFIG.QDRANT_DOC_COLLECTION;
    const url = `${qdrantUrl}/collections/${collectionName}/index`;

    try {
        // This creates an index on the 'content_hash' field for fast lookups.
        await axios.put(url, { field_name: fieldName, field_schema: "keyword" }, { headers });
        console.log(`[Indexer] ✅ Ensured payload index exists for 'content_hash' in ${collectionName}.`);
    } catch (error) {
        if (error.response?.status === 409) { // 409 Conflict means it already exists
            // This is expected on subsequent runs, so we don't log an error.
        } else {
            console.error(`[Indexer] ⚠️ Could not create Qdrant payload index:`, error.response?.data || error.message);
        }
    }
}


/**
 * Modified point creation functions to include the hash.
 */
function createChunkPoints(fileId, fileName, chunks, openaiEmbeddings, geminiEmbeddings, summaries, fileHash) {
    return chunks.map((chunk, i) => {
        const basePayload = {
            // ... all previous fields
            content_hash: fileHash // NEW
        };
        const payload = normalizePayload(basePayload);
        return { id: crypto.randomUUID(), vector: { /*...*/ }, payload };
    });
}

function createDocPoint(fileId, fileName, chunks, openaiEmbeddings, geminiEmbeddings, docSummary, fileHash) {
    const basePayload = {
        // ... all previous fields
        content_hash: fileHash // NEW
    };
    const payload = normalizePayload(basePayload);
    delete payload.chunk_text;
    return { id: crypto.randomUUID(), vector: { /*...*/ }, payload };
}

// ... PASTE ALL OTHER PREVIOUSLY DEFINED HELPER FUNCTIONS HERE ...
// (getApiKeys, getTextFromImageOrPdf, normalizePayload, etc.)
