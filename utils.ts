import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAI } from 'langchain/llms/openai';
import { loadQAStuffChain } from 'langchain/chains';
import { Document } from 'langchain/document';
import { timeout } from './config';

export const createPineconeIndex = async (client, indexName, vectorDimension) => {
    // 1. Initiate index existence check
    console.log(`Checking "${indexName}"...`);
    // 2. Get list of existing indexes
    const existingIndexes = await client.listIndexes();
    // 3. If index doesn't exist, create it
    if (!existingIndexes.includes(indexName)) {
        // 4. Log index creation initiation
        console.log(`Creating "${indexName}"...`);
        // 5. Create index
        await client.createIndex({
            createRequest: {
                name: indexName,
                dimension: vectorDimension,
                metric: 'cosine',
            },
        });
        console.log(`Index creation initiated. Please wait...`);
        await new Promise((resolve) => setTimeout(resolve, timeout));
    } else {
        console.log(`"${indexName}" already exists`);
    }
};

export const updatePinecone = async (client, indexName, docs) => {
    const index = client.index(indexName);
    console.log(`Pinecone index retrieved: ${indexName}`);

    for (const doc of docs) {
        const txtPath = doc.metadata.source;
        const text = doc.pageContent;
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
        });

        const chunks = await textSplitter.create([text]);
        const embeddingsArray = await new OpenAIEmbeddings().embedDocuments(
            chunks.map((chunk) => chunk.pageContent.replace(/\n/g, " "))
        );
        console.log('Finished embedding documents');
        console.log(
            `Creating ${chunks.length} vectors array with id, values, and metadata...`
        );

        const batchSize = 100;
        let batch = [];

        for (let idx = 0; idx < chunks.length; idx++) {
            const chunk = chunks[idx];
            const vector = {
                id: `${txtPath}_${idx}`,
                values: embeddingsArray[idx],
                metadata: {
                    ...chunk.metadata,
                    loc: JSON.stringify(chunk.metadata.loc),
                    pageContent: chunk.pageContent,
                    txtPath: txtPath,
                },
            };
            batch.push(vector);

            // When batch is full or it's the last item, upsert the vectors
            if (batch.length === batchSize || idx === chunks.length - 1) {
                await index.upsert({
                    upsertRequest: {
                        vectors: batch,
                    },
                });
                // Empty the batch
                batch = [];
            }
        }
    }
};

export const queryPineconeVectorStoreAndQueryLLM = async (client, indexName, question) => {
    const index = client.index(indexName);
    const queryEmbeddings = await new OpenAIEmbeddings().embedQuery(question);
    
    let queryResponse = await index.query({
        queryRequest: {
            topK: 10,
            vector: queryEmbeddings,
            includeMetadata: true,
            includeValues: true,
        }
    });

    console.log(`Found ${queryResponse.matches.length} matches...`);
    console.log(`Asking question: ${question}...`);

    if (queryResponse.matches.length) {
        const llm = new OpenAI({});
        const chain = loadQAStuffChain(llm);

        const concatenatedPageContent = queryResponse.matches
            .map((match) => match.metadata.pageContent)
            .join(" ");
            
        const result = await chain.call({
            input_documents: [new Document({ pageContent: concatenatedPageContent })],
            question: question,
        });

        console.log(`Answer: ${result.text}`);
        return result.text;
    } else {
        console.log('Since there are no matches, GPT-3 will not be queried.');
    }
};
