# TPU Bench IDE Implementation

You are an expert full-stack engineer. Your task is to implement the JAX-on-TPU Development Environment as specified in the Product Requirements Document (PRD_JAX_TPU_Tool.md).

You must use the following technical stack and architecture:

Frontend: React (as a single .jsx file), using Tailwind CSS.

Backend & Database: Firebase (Firebase Auth, Firestore, Firebase Functions).

Compute Backend: Google Kubernetes Engine (GKE) with TPU node pools.

1. Frontend Implementation (React App)

The user interface must be clean, modern, and contain three main sections:

Header:

App Title ("JAX TPU Lab").

Authentication button (using Firebase Auth, showing "Sign In" or "Welcome, user@email.com").

Editor & Tabs (Main Section):

Use the Monaco Editor for code editing.

Implement a tabbed interface:

main.py: A read/write tab containing the Monaco Editor for the user's JAX code.

README.md: A read-only tab displaying the problem description (use placeholder Markdown).

requirements.txt: A read/write tab for Python package dependencies.

Implement action buttons:

"Review Code" (Gemini): This button triggers the Gemini Code Review workflow.

"Run on TPU": This button triggers the Async Job Execution workflow.

Results & Logs (Bottom Panel):

This panel must display the real-time status of a job.

It must listen to the active job document in Firestore using onSnapshot.

It should display:

Job Status: (e.g., QUEUED, COMPILING, RUNNING, SUCCESS, FAILED).

Logs: Real-time log output (stdout/stderr) streamed from the job.

Performance: Final metrics (execution time, etc.) when the job is complete.

2. Backend (Firebase)

2.1 Firebase Auth

Implement Google Sign-In for user authentication. All user data and jobs must be tied to their userId.

2.2 Firestore Database Structure

Use Firestore to manage state. Enforce strict security rules.

users/{userId}/code/{docId}: Stores the user's code files (main.py, requirements.txt).

users/{userId}/jobs/{jobId}: Stores the state for each execution. This is the "source of truth" for the UI.

status: (string) "QUEUED", "RUNNING", "COMPLETED", "FAILED"

code: (string) The main.py code at the time of execution.

logs: (string) Real-time log output.

metrics: (map) { compilationTime, executionTime, accuracy }

createdAt: (timestamp)

2.3 Firebase Functions (Control Plane API)

You will create two main HTTP-triggered Firebase Functions:

reviewCode(code: string):

Takes the user's code as input.

Constructs a system prompt: "You are an expert in JAX and TPU optimization. Review this code and provide feedback on optimization gaps (missing @jit, pmap), potential bugs, and best practices."

Calls the Gemini API (generateContent).

Returns the AI-generated review as a Markdown string.

runJob(code: string, requirements: string):

This is the async job trigger. It must not run the code.

Gets the authenticated userId.

Creates a new document in Firestore: users/{userId}/jobs/{jobId} with status: "QUEUED".

Uses the Google Cloud GKE API to programmatically create a new Kubernetes Job.

The K8s Job spec must:

Point to a custom Docker image (see below).

Request TPU resources (e.g., google.com/tpu).

Pass the jobId and userId as environment variables.

Returns the new jobId to the frontend immediately.

3. Compute (GKE Job Execution)

3.1 Dockerfile (JAX Runner)

Create a Dockerfile based on a JAX/TPU-ready base image.

It must install firebase-admin and other dependencies.

The ENTRYPOINT should be a Python script (runner.py).

3.2 runner.py (Inside the Pod)

This script is the core of the execution and runs inside the GKE pod.

Initialize:

Reads JOB_ID and USER_ID from environment variables.

Initializes the firebase-admin SDK.

Gets the job document: docRef = db.collection('users').doc(USER_ID).collection('jobs').doc(JOB_ID).

Run & Stream Logs:

Fetches the user's code and requirements.txt from the docRef.

Updates job status: docRef.update({ status: "COMPILING" }).

Installs user-defined requirements.

Executes the user's main.py code as a subprocess.

Crucially: It must capture stdout and stderr from the subprocess in real-time and append them to the logs field in the Firestore document, allowing the user to see live output.

Updates status: docRef.update({ status: "RUNNING" }).

Finish:

When the subprocess completes, it captures the final exit code.

Parses the logs for key metrics (e.g., "Final accuracy: 0.92").

Updates the job document one last time:

status: "COMPLETED" (or "FAILED")

metrics: { ... }

finishedAt: (timestamp)