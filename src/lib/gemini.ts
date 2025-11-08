import { VertexAI } from '@google-cloud/vertexai';

const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT_ID;
const REGION = process.env.GOOGLE_CLOUD_REGION;

if (!PROJECT_ID || !REGION) {
  console.error("Please set the GOOGLE_CLOUD_PROJECT_ID and GOOGLE_CLOUD_REGION environment variables.");
}

const vertexAI = new VertexAI({ project: PROJECT_ID, location: REGION });
const model = 'gemini-1.5-flash-001';

const generativeModel = vertexAI.getGenerativeModel({
  model: model,
});

export async function callGemini(prompt: string): Promise<string> {
  if (!PROJECT_ID || !REGION) {
    return "Please set the GOOGLE_CLOUD_PROJECT_ID and GOOGLE_CLOUD_REGION environment variables.";
  }
  try {
    const resp = await generativeModel.generateContent(prompt);
    const content = resp.response.candidates[0].content.parts[0].text;
    return content || "No content generated.";
  } catch (error) {
    console.error(error);
    if (error instanceof Error) {
      return `Error calling Gemini API: ${error.message}`;
    }
    return "An unknown error occurred while calling the Gemini API.";
  }
}
