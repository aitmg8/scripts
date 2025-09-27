// =================================================================================
// IMPORTS AND CLIENT INITIALIZATION
// =================================================================================
const { SecretManagerServiceClient } = require('@google-cloud/secret-manager');
const { Storage } = require('@google-cloud/storage');
const { ImageAnnotatorClient } = require('@google-cloud/vision');
const axios = require('axios');
const crypto = require('crypto');

const storage = new Storage();
const secretManager = new SecretManagerServiceClient();
const visionClient = new ImageAnnotatorClient();

// =================================================================================
// CONFIGURATION (Mirrors Apps Script Globals)
// =================================================================================
const WF_CONFIG = {
    // Buckets
    COMPLETED_BUCKET_NAME: 'files-completed', // CHANGE if needed
    ERROR_BUCKET_NAME: 'files-error',       // CHANGE if needed

    // Qdrant Collections
    QDRANT_CHUNK_COLLECTION: 'nexus_shards',
    QDRANT_DOC_COLLECTION: 'cortex_nodes',

    // AI Models
    EMBEDDING_MODEL_OPENAI: 'text-embedding-3-small',
    EMBEDDING_MODEL_GEMINI: 'embedding-001', // Corresponds to text-embedding-004 in GAS
    SUMMARY_MODEL: 'gpt-4o-mini',

    // Processing Parameters
    CHUNK_SIZE: 500, // In words
    CHUNK_OVERLAP: 50, // In words
    SUMMARY_TEMPERATURE: 0.2,
    SUMMARY_MAX_TOKENS: 1400,

    // Normalization
    NORMALIZER_VERSION: "3.2.0",

    // Categories & Subtypes for Classification
    CATEGORIES: ["Finance", "Lodging", "Health", "Career", "Travel", "Hobbies", "Vital Records"],
    SUBTYPES: [
        "Lease-Agreement", "Lease-Addendum", "Lease-Renewal", "Guarantor-Form", "Move-In-Checklist", "Move-Out-Checklist",
        "Rent-Statement", "Rent-Receipt", "Invoice", "Receipt", "Quote", "Estimate", "SD-Statement", "Utility-Bill",
        "Notice", "Letter", "Complaint", "Insurance-Policy", "Property-Tax-Bill", "HOA-Letter", "HOA-Notice",
        "Statement", "Bill", "Payslip", "Confirmation", "Loan-Statement", "Tax-Return", "Tax-Form", "Tax-Payment",
        "Employment-Contract", "Offer-Letter", "Expense-Report", "Policy", "Premium-Bill", "Claim", "EOB",
        "Leistungsabrechnung", "Medical-Bill", "ID-Scan", "Passport-Scan", "Permit-Scan", "Registration",
        "Vertrag", "Contract", "Warranty", "Will"
    ],
    SUBTYPE_TO_CATEGORY: {
        "Lease-Agreement": "Lodging", "Lease-Addendum": "Lodging", "Lease-Renewal": "Lodging", "Guarantor-Form": "Lodging",
        "Move-In-Checklist": "Lodging", "Move-Out-Checklist": "Lodging", "Rent-Statement": "Lodging", "Rent-Receipt": "Lodging",
        "HOA-Letter": "Lodging", "HOA-Notice": "Lodging", "Property-Tax-Bill": "Lodging", "Utility-Bill": "Lodging",
        "Notice": "Lodging", "Complaint": "Lodging",
        "Invoice": "Finance", "Receipt": "Finance", "Statement": "Finance", "Bill": "Finance", "Loan-Statement": "Finance",
        "Tax-Return": "Finance", "Tax-Form": "Finance", "Tax-Payment": "Finance", "Expense-Report": "Finance",
        "Policy": "Finance", "Premium-Bill": "Finance", "Quote": "Finance", "Estimate": "Finance", "Warranty": "Finance",
        "Medical-Bill": "Health", "EOB": "Health", "Leistungsabrechnung": "Health", "Claim": "Health",
        "Employment-Contract": "Career", "Offer-Letter": "Career", "Payslip": "Career", "Contract": "Career", "Vertrag": "Career",
        "ID-Scan": "Vital Records", "Passport-Scan": "Vital Records", "Permit-Scan": "Vital Records",
        "Registration": "Vital Records", "Will": "Vital Records"
    },
    CATEGORY_HINTS: {
        "Finance": ["invoice", "receipt", "statement", "bill", "tax", "irs", "loan", "bank", "payment", "premium", "policy", "warranty", "quote", "estimate", "expense", "credit", "debit", "account", "balance", "transfer", "refund"],
        "Lodging": ["tenant", "lease", "rent", "move-in", "move out", "move-out", "guarantor", "hoa", "security deposit", "property tax", "utility", "internet", "water", "gas", "electric", "eviction", "notice", "complaint", "apartment", "unit", "landlord", "residence", "property"],
        "Health": ["medical", "hospital", "clinic", "eob", "benefits", "health insurance", "provider", "diagnosis", "procedure", "icd", "claim", "copay", "deductible", "leistungsabrechnung"],
        "Career": ["employment", "employee", "employer", "offer letter", "hire", "onboarding", "contract", "payslip", "pay stub", "salary", "compensation", "hr", "position", "role", "job"],
        "Travel": ["itinerary", "boarding pass", "pnr", "confirmation code", "booking", "reservation", "flight", "airport", "iata", "hotel", "check-in", "check out", "car rental", "airline", "gate", "terminal", "visa"],
        "Hobbies": ["membership", "club", "course", "class", "event", "ticket", "competition", "gym", "workshop", "training", "hobby", "league", "tournament", "subscription", "festival"],
        "Vital Records": ["passport", "identity", "id", "driver license", "permit", "residence permit", "registration", "birth", "marriage", "death", "notary", "will", "testament", "document copy", "scan", "certificate"]
    }
};

// =================================================================================
// NORMALIZATION & CLASSIFICATION ENGINE (Ported from Apps Script)
// =================================================================================

function cleanText(s) {
    if (typeof s !== 'string') return '';
    return s.replace(/[^\x09\x0A\x0D\x20-\x7E]/g, '') // Keep only printable ASCII + whitespace
        .replace(/[ \t]+/g, ' ')
        .replace(/\s+\n/g, '\n')
        .replace(/\n\s+/g, '\n')
        .trim();
}

function guessSubtypeFromText(fileName, content) {
    const hay = [fileName, content].join(' ').toLowerCase();
    const map = [
        ["Lease-Agreement", /lease[_\s-]?agreement/], ["Lease-Addendum", /lease[_\s-]?addendum/], ["Lease-Renewal", /lease[_\s-]?renew(al|)/],
        ["Guarantor-Form", /guarantor|co[-\s]?sign/], ["Move-In-Checklist", /move[-\s]?in[_\s-]?checklist/], ["Move-Out-Checklist", /move[-\s]?out[_\s-]?checklist/],
        ["Rent-Statement", /rent[_\s-]?statement/], ["Rent-Receipt", /rent[_\s-]?receipt/], ["Invoice", /\binvoice\b|(^|[\s_])inv[-_\s]?\d{2,}/],
        ["Receipt", /\breceipt\b/], ["Quote", /\bquote\b/], ["Estimate", /\bestimate\b/], ["SD-Statement", /(security\s+deposit|sd[-\s]?statement)/],
        ["Utility-Bill", /(utility|internet|water|gas|electric).{0,12}\bbill\b/], ["Notice", /\bnotice\b/], ["Letter", /\b(dear\s+\w+|letter)\b/],
        ["Complaint", /\bcomplaint\b/], ["Insurance-Policy", /\binsurance\s+policy\b/], ["Property-Tax-Bill", /\bproperty\s+tax\b/],
        ["HOA-Letter", /\bhoa\b.*\bletter\b/], ["HOA-Notice", /\bhoa\b.*\bnotice\b/], ["Statement", /\bstatement\b/], ["Bill", /\bbill\b/],
        ["Payslip", /\bpayslip\b|pay\s*stub|salary\s*slip/], ["Confirmation", /\bconfirmation\b/],
        ["Loan-Statement", /\b(loan|mortgage|credit)\s+statement\b/], ["Tax-Return", /\btax\s+return\b|form\s+10(40|65)\b|tax\s+declaration/],
        ["Tax-Form", /\b(w-2|1099|k-1|tax\s+form)\b/i], ["Tax-Payment", /\btax\s+payment\b|irs\s+payment/],
        ["Employment-Contract", /\bemployment\s+contract\b/], ["Offer-Letter", /\boffer\s+letter\b/], ["Expense-Report", /\bexpense\s+report\b/],
        ["Policy", /\bpolicy\b/], ["Premium-Bill", /\bpremium\s+bill\b/], ["Claim", /\bclaim\b|insurance\s+claim/],
        ["EOB", /\bexplanation\s+of\s+benefits\b|\beob\b/], ["Leistungsabrechnung", /\bleistungsabrechnung\b/],
        ["Medical-Bill", /\bmedical\s+bill\b|hospital\s+bill|clinic\s+bill/],
        ["ID-Scan", /\b(id|identity|driver.?s?\s*license)\b.*(scan|copy)|\bpassport\b.*(scan|copy)/],
        ["Passport-Scan", /\bpassport\b.*(scan|copy)/], ["Permit-Scan", /\b(work|residence)\s+permit\b|\bpermit\s+scan\b/],
        ["Registration", /\bregistration\b/], ["Vertrag", /\bvertrag\b/], ["Contract", /\bcontract\b|\bagreement\b/], ["Warranty", /\bwarranty\b/],
        ["Will", /\b(last\s+will|testament)\b/]
    ];
    for (const [subtype, regex] of map) {
        if (regex.test(hay)) return subtype;
    }
    return null;
}

function classifyPayload(payload) {
    const searchText = [payload.file_name, payload.chunk_text].join(' ').toLowerCase();
    const guessedSubtype = guessSubtypeFromText(payload.file_name, searchText);

    let finalCategory = "Finance"; // Default
    if (guessedSubtype && WF_CONFIG.SUBTYPE_TO_CATEGORY[guessedSubtype]) {
        finalCategory = WF_CONFIG.SUBTYPE_TO_CATEGORY[guessedSubtype];
    }

    let bestCat = finalCategory;
    let bestScore = 0;
    for (const cat in WF_CONFIG.CATEGORY_HINTS) {
        let score = WF_CONFIG.CATEGORY_HINTS[cat].reduce((acc, hint) => acc + (searchText.includes(hint) ? 1 : 0), 0);
        if (cat === finalCategory) score += 3; // Boost score for matching subtype's category

        if (score > bestScore) {
            bestScore = score;
            bestCat = cat;
        }
    }
    return {
        subtype: guessedSubtype,
        category: bestCat,
        classification_confidence: Number(Math.min(1.0, bestScore / 5.0).toFixed(2))
    };
}

function normalizePayload(payload) {
    const patch = {};
    const classification = classifyPayload(payload);
    patch.category = classification.category;
    patch.subtype = classification.subtype;
    patch.classification_confidence = classification.classification_confidence;
    
    // Add other normalization logic from GAS here if needed (e.g., date parsing, phone numbers)
    // For brevity, this example focuses on the core classification.

    patch.normalized_at = new Date().toISOString();
    patch.normalizer_version = WF_CONFIG.NORMALIZER_VERSION;
    
    return { ...payload, ...patch };
}

// =================================================================================
// API AND UTILITY HELPERS
// =================================================================================

async function getApiKeys(projectId) {
    const secrets = [
        'OPENAI_API_KEY', 'GEMINI_API_KEY', 'QDRANT_API_KEY', 'QDRANT_API_URL'
    ];
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

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function backoffAttempt(fn, label = 'operation', maxRetries = 5, initialDelay = 1000) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            const delay = initialDelay * Math.pow(2, i) * (1 + Math.random() * 0.25);
            console.warn(`⏳ ${label} failed. Retrying in ${delay.toFixed(0)}ms...`, error.message);
            await sleep(delay);
        }
    }
}

/**
 * Extracts text from images or PDFs using Google Cloud Vision API.
 * @param {string} bucketName The GCS bucket name.
 * @param {string} fileName The file name.
 * @returns {Promise<string>} The extracted text.
 */
async function getTextFromImageOrPdf(bucketName, fileName) {
    console.log(`👁️ Performing OCR on gs://${bucketName}/${fileName}`);
    const [result] = await visionClient.textDetection(`gs://${bucketName}/${fileName}`);
    const detection = result.fullTextAnnotation;
    return detection ? detection.text : '';
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

// =================================================================================
// AI EMBEDDING AND SUMMARY FUNCTIONS
// =================================================================================

async function getOpenAIEmbeddings(chunks, apiKey) {
    return backoffAttempt(async () => {
        const response = await axios.post('https://api.openai.com/v1/embeddings', {
            input: chunks, model: WF_CONFIG.EMBEDDING_MODEL_OPENAI
        }, { headers: { 'Authorization': `Bearer ${apiKey}` } });
        return response.data.data.map(item => item.embedding);
    }, 'OpenAI Embeddings');
}

async function getGeminiEmbeddings(chunks, apiKey) {
    return backoffAttempt(async () => {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${WF_CONFIG.EMBEDDING_MODEL_GEMINI}:batchEmbedText?key=${apiKey}`;
        const requests = chunks.map(chunk => ({ model: `models/${WF_CONFIG.EMBEDDING_MODEL_GEMINI}`, content: { parts: [{ text: chunk }] } }));
        const response = await axios.post(url, { requests });
        return response.data.embeddings.map(item => item.values);
    }, 'Gemini Embeddings');
}

async function getChunkSummaries(chunks, apiKey) {
    const promises = chunks.map(chunk => backoffAttempt(async () => {
        const response = await axios.post('https://api.openai.com/v1/chat/completions', {
            model: WF_CONFIG.SUMMARY_MODEL,
            messages: [
                { role: 'system', content: 'You are a meticulous document analyst. Summarize the following text, capturing all key details, figures, and names precisely.' },
                { role: 'user', content: chunk }
            ],
            temperature: WF_CONFIG.SUMMARY_TEMPERATURE, max_tokens: WF_CONFIG.SUMMARY_MAX_TOKENS,
        }, { headers: { 'Authorization': `Bearer ${apiKey}` } });
        return response.data.choices[0].message.content.trim();
    }, 'Chunk Summary'));
    return Promise.all(promises);
}

async function getDocSummary(chunkSummaries, apiKey) {
    const topSummaries = chunkSummaries
        .map((s, i) => ({ text: s, len: s.length, i }))
        .sort((a, b) => b.len - a.len)
        .slice(0, 12)
        .map(item => `--- CHUNK ${item.i} ---\n${item.text}`)
        .join("\n\n");

    const prompt = `Combine the following chunk summaries into a single, comprehensive document-level summary. Provide:\n- Executive overview\n- Key entities and roles\n- Important dates & deadlines\n- Amounts & currencies\n- IDs/References\n- Terms & obligations\n- Actions required\n\n${topSummaries}`;
    
    return backoffAttempt(async () => {
        const response = await axios.post('https://api.openai.com/v1/chat/completions', {
            model: WF_CONFIG.SUMMARY_MODEL,
            messages: [
                { role: 'system', content: 'You are a senior analyst. Write a comprehensive, well-structured document-level summary combining multiple chunk summaries.' },
                { role: 'user', content: prompt }
            ],
            temperature: WF_CONFIG.SUMMARY_TEMPERATURE, max_tokens: WF_CONFIG.SUMMARY_MAX_TOKENS
        }, { headers: { 'Authorization': `Bearer ${apiKey}` } });
        return response.data.choices[0].message.content.trim();
    }, 'Doc Summary');
}

// =================================================================================
// QDRANT INTERACTION
// =================================================================================

async function upsertPoints(points, collectionName, { qdrantUrl, qdrantKey }) {
    if (points.length === 0) return;
    const url = `${qdrantUrl}/collections/${collectionName}/points?wait=true`;
    console.log(`🚚 Upserting ${points.length} points to ${collectionName}`);
    await backoffAttempt(() => axios.put(url, { points }, {
        headers: { 'api-key': qdrantKey, 'Content-Type': 'application/json' }
    }), `Qdrant Upsert to ${collectionName}`);
}

async function deletePointsByFileId(fileId, collectionName, { qdrantUrl, qdrantKey }) {
    const url = `${qdrantUrl}/collections/${collectionName}/points/delete?wait=true`;
    console.log(`🧹 Deleting existing points for file ID ${fileId} from ${collectionName}`);
    const payload = { filter: { must: [{ key: "file_id", match: { value: fileId } }] } };
    await backoffAttempt(() => axios.post(url, payload, {
        headers: { 'api-key': qdrantKey, 'Content-Type': 'application/json' }
    }), `Qdrant Delete from ${collectionName}`);
}

// =================================================================================
// POINT CREATION (Chunk and Document Level)
// =================================================================================

function createChunkPoints(fileId, fileName, chunks, openaiEmbeddings, geminiEmbeddings, summaries) {
    return chunks.map((chunk, i) => {
        const basePayload = {
            file_id: fileId,
            file_name: fileName,
            source_uri: `gs://${fileId}`, // Using fileId as a placeholder for full path
            chunk_text: chunk,
            ai_summary: summaries[i],
            chunk_index: i,
            total_chunks: chunks.length,
            processed_at: new Date().toISOString()
        };
        const payload = normalizePayload(basePayload);
        
        return {
            id: crypto.randomUUID(),
            vector: {
                openai: openaiEmbeddings[i],
                gemini: geminiEmbeddings[i]
            },
            payload
        };
    });
}

function createDocPoint(fileId, fileName, chunks, openaiEmbeddings, geminiEmbeddings, docSummary) {
    const meanVec = (arr) => {
        if (!arr || !arr.length) return [];
        const dim = arr[0].length;
        const sum = new Array(dim).fill(0);
        for (const vec of arr) {
            for (let i = 0; i < dim; i++) sum[i] += vec[i];
        }
        return sum.map(v => v / arr.length);
    };

    // Use first chunk for classification context
    const basePayload = {
        file_id: fileId,
        file_name: fileName,
        source_uri: `gs://${fileId}`,
        chunk_text: chunks[0] || '', // Temp text for classification
        doc_summary: docSummary,
        chunk_count: chunks.length,
        processed_at: new Date().toISOString()
    };
    
    const payload = normalizePayload(basePayload);
    delete payload.chunk_text; // Remove temp text after normalization

    return {
        id: crypto.randomUUID(),
        vector: {
            openai: meanVec(openaiEmbeddings),
            gemini: meanVec(geminiEmbeddings)
        },
        payload
    };
}


// =================================================================================
// MAIN CLOUD FUNCTION
// =================================================================================

exports.processFile = async (file, context) => {
    const projectId = process.env.GCP_PROJECT;
    const sourceBucketName = file.bucket;
    const sourceFileName = file.name;
    const fileId = `${sourceBucketName}/${sourceFileName}`; // Unique ID

    console.log(`🚀 Starting processing for: ${fileId}`);

    try {
        // 1. Get secrets
        const apiKeys = await getApiKeys(projectId);

        // 2. Extract text (with OCR for images/PDFs)
        let textContent;
        if (file.contentType?.startsWith('image/') || file.contentType === 'application/pdf') {
            textContent = await getTextFromImageOrPdf(sourceBucketName, sourceFileName);
        } else {
            const buffer = await storage.bucket(sourceBucketName).file(sourceFileName).download();
            textContent = buffer.toString();
        }

        if (!textContent || !textContent.trim()) {
            console.log(`⏭️ Skipping empty file: ${sourceFileName}`);
            // Optionally move to a 'skipped' bucket
            return;
        }

        // 3. Chunk text
        const chunks = chunkText(textContent);
        console.log(`🧩 Created ${chunks.length} chunks.`);

        // 4. Generate embeddings and summaries in parallel
        console.log("🧠 Generating embeddings and summaries...");
        const [openaiEmbeddings, geminiEmbeddings, chunkSummaries] = await Promise.all([
            getOpenAIEmbeddings(chunks, apiKeys.openAIKey),
            getGeminiEmbeddings(chunks, apiKeys.geminiKey),
            getChunkSummaries(chunks, apiKeys.openAIKey)
        ]);

        // 5. Generate document-level summary
        console.log("📝 Generating document-level summary...");
        const docSummary = await getDocSummary(chunkSummaries, apiKeys.openAIKey);

        // 6. Create points for both collections
        const chunkPoints = createChunkPoints(fileId, sourceFileName, chunks, openaiEmbeddings, geminiEmbeddings, chunkSummaries);
        const docPoint = createDocPoint(fileId, sourceFileName, chunks, openaiEmbeddings, geminiEmbeddings, docSummary);
        
        // 7. Delete existing points for this file to ensure idempotency
        await Promise.all([
            deletePointsByFileId(fileId, WF_CONFIG.QDRANT_CHUNK_COLLECTION, apiKeys),
            deletePointsByFileId(fileId, WF_CONFIG.QDRANT_DOC_COLLECTION, apiKeys)
        ]);

        // 8. Upsert new points to Qdrant
        await Promise.all([
            upsertPoints(chunkPoints, WF_CONFIG.QDRANT_CHUNK_COLLECTION, apiKeys),
            upsertPoints([docPoint], WF_CONFIG.QDRANT_DOC_COLLECTION, apiKeys)
        ]);
        
        // 9. Move file to 'completed' bucket
        await storage.bucket(sourceBucketName).file(sourceFileName).move(
            storage.bucket(WF_CONFIG.COMPLETED_BUCKET_NAME).file(sourceFileName)
        );

        console.log(`✅ Successfully processed and moved ${sourceFileName}.`);

    } catch (error) {
        console.error(`❌ Failed to process file ${sourceFileName}:`, error.response ? error.response.data : error);
        // Move file to 'error' bucket for review
        try {
            await storage.bucket(sourceBucketName).file(sourceFileName).move(
                storage.bucket(WF_CONFIG.ERROR_BUCKET_NAME).file(sourceFileName)
            );
            console.log(`📁 Moved failed file to error bucket.`);
        } catch (moveError) {
            console.error(`‼️ Failed to move error file:`, moveError);
        }
    }
};
