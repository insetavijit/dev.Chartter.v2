#!/usr/bin/env node

const simpleGit = require("simple-git");
const git = simpleGit();

// Generate timestamp for commit message
const now = new Date();
const timestamp = now.toISOString(); // e.g., "2025-09-20T12:34:56.789Z"

async function autoCommit() {
  try {
    // Initialize repo if not exists
    if (!(await git.checkIsRepo())) {
      console.log("Initializing Git repository...");
      await git.init();
    }

    // Stage all changes
    console.log("Adding all changes...");
    await git.add("./*");

    // Commit with timestamp
    console.log(`Committing: "${timestamp}"`);
    await git.commit(timestamp);

    // Push to remote main branch
    console.log("Pushing to origin/main...");
    await git.push("origin", "main", { "--set-upstream": null });

    console.log("✅ Auto commit and push completed!");
  } catch (err) {
    console.error("Git automation failed:", err);
  }
}

autoCommit();
