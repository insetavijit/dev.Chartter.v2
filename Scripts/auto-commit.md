Here's an improved version of your Node.js script for automating Git commits, with added features, better error handling, and enhanced configurability. The improvements include command-line argument support, branch selection, custom commit messages, and safer Git operations.

<xaiArtifact artifact_id="a4d5e89f-1580-47ca-a57d-944875bd5623" artifact_version_id="32053490-2543-46a5-b36c-a1b842184169" title="auto-commit.js" contentType="text/javascript">
#!/usr/bin/env node

const simpleGit = require('simple-git');
const yargs = require('yargs');
const git = simpleGit();

// Parse command-line arguments
const argv = yargs
  .option('branch', {
    alias: 'b',
    type: 'string',
    default: 'main',
    description: 'Branch to commit and push to',
  })
  .option('message', {
    alias: 'm',
    type: 'string',
    description: 'Custom commit message (overrides timestamp)',
  })
  .option('remote', {
    alias: 'r',
    type: 'string',
    default: 'origin',
    description: 'Remote repository name',
  })
  .option('dry-run', {
    type: 'boolean',
    default: false,
    description: 'Simulate the Git operations without executing them',
  })
  .help()
  .argv;

// Generate timestamp for default commit message
const now = new Date();
const timestamp = now.toISOString().replace(/[:.]/g, '-'); // e.g., 2025-09-20T12-34-56-789Z
const commitMessage = argv.message || `Auto-commit: ${timestamp}`;

// Logger for consistent output
const log = (message, isError = false) => {
  const prefix = isError ? '❌' : 'ℹ️';
  console.log(`${prefix} ${message}`);
};

async function autoCommit() {
  try {
    // Check if in a Git repository
    if (!(await git.checkIsRepo())) {
      log('Not a Git repository. Initializing...');
      if (!argv.dryRun) {
        await git.init();
        log('Git repository initialized');
      } else {
        log('Would initialize Git repository (dry run)');
      }
    }

    // Check for changes
    const status = await git.status();
    if (!status.files.length) {
      log('No changes to commit');
      return;
    }

    // Stage all changes
    log('Adding all changes...');
    if (!argv.dryRun) {
      await git.add('./*');
    } else {
      log('Would stage all changes (dry run)');
    }

    // Commit changes
    log(`Committing with message: "${commitMessage}"`);
    if (!argv.dryRun) {
      await git.commit(commitMessage);
    } else {
      log('Would commit changes (dry run)');
    }

    // Check if remote exists
    const remotes = await git.getRemotes(true);
    const remoteExists = remotes.some((r) => r.name === argv.remote);
    if (!remoteExists) {
      log(`Remote "${argv.remote}" not found. Skipping push.`);
      return;
    }

    // Ensure branch exists locally
    const branches = await git.branchLocal();
    if (!branches.all.includes(argv.branch)) {
      log(`Branch "${argv.branch}" does not exist locally. Creating...`);
      if (!argv.dryRun) {
        await git.checkoutLocalBranch(argv.branch);
      } else {
        log(`Would create branch "${argv.branch}" (dry run)`);
      }
    }

    // Push to remote
    log(`Pushing to ${argv.remote}/${argv.branch}...`);
    if (!argv.dryRun) {
      await git.push(argv.remote, argv.branch, { '--set-upstream': null });
    } else {
      log(`Would push to ${argv.remote}/${argv.branch} (dry run)`);
    }

    log('✅ Auto commit and push completed!', false);
  } catch (err) {
    log(`Git automation failed: ${err.message}`, true);
    process.exit(1);
  }
}

// Run the script
autoCommit();
</xaiArtifact>

### Improvements Made:

1. **Command-Line Arguments**:
   - Added `yargs` for parsing command-line arguments, allowing users to:
     - Specify the branch to commit and push to (`--branch` or `-b`).
     - Provide a custom commit message (`--message` or `-m`).
     - Specify the remote repository name (`--remote` or `-r`).
     - Enable a dry-run mode (`--dry-run`) to simulate operations without executing them.
   - Example usage:
     ```bash
     node auto-commit.js --branch develop --message "Updated files" --remote origin
     node auto-commit.js --dry-run
     ```

2. **Better Timestamp Format**:
   - Replaced colons and periods in the timestamp with hyphens (e.g., `2025-09-20T12-34-56-789Z`) for better readability and file system compatibility.

3. **Improved Error Handling**:
   - Added try-catch blocks with specific error messages for better debugging.
   - Exits with a non-zero code on failure for scripting integration.
   - Checks for the existence of the remote and branch before pushing.

4. **No-Op for No Changes**:
   - Checks the Git status to avoid creating empty commits if there are no changes to stage.

5. **Dry-Run Mode**:
   - Added a `--dry-run` flag to simulate Git operations without making actual changes, useful for testing.

6. **Branch Management**:
   - Automatically creates the specified branch locally if it doesn't exist.
   - Ensures the branch is set up to track the remote branch during push.

7. **Improved Logging**:
   - Introduced a `log` function for consistent output with emoji indicators (ℹ️ for info, ❌ for errors).
   - Provides clear feedback for each step, including dry-run messages.
   - Displays the commit message and branch details in logs.

8. **Dependency Addition**:
   - Added `yargs` as a dependency for command-line parsing. Install it with:
     ```bash
     npm install yargs simple-git
     ```

### How to Use:
1. Save the script as `auto-commit.js`.
2. Install the required dependencies:
   ```bash
   npm install simple-git yargs
   ```
3. Run the script with various options:
   ```bash
   # Default: Commit and push to origin/main with timestamp
   node auto-commit.js

   # Custom branch
   node auto-commit.js --branch develop

   # Custom commit message
   node auto-commit.js --message "Updated project files"

   # Custom remote
   node auto-commit.js --remote upstream

   # Dry run to simulate
   node auto-commit.js --dry-run

   # Combine options
   node auto-commit.js --branch feature --message "Feature complete" --remote origin --dry-run
   ```

### Example Output:
```bash
ℹ️ Adding all changes...
ℹ️ Committing with message: "Auto-commit: 2025-09-20T11-18-00-000Z"
ℹ️ Pushing to origin/main...
✅ Auto commit and push completed!
```

### Additional Suggestions:
- **Config File Support**: Allow defaults (e.g., branch, remote, message template) to be set in a `.git-auto-commit.json` file.
- **Exclude Files**: Add an option to exclude certain files or patterns from being staged (e.g., using `.gitignore` or a custom pattern).
- **Pre-Commit Checks**: Run linting or tests before committing (e.g., integrate with `eslint` or a test runner).
- **Multiple Remotes**: Support pushing to multiple remotes simultaneously.
- **Commit Message Templates**: Allow users to define a commit message template with placeholders (e.g., `{timestamp}`, `{branch}`).

Let me know if you'd like to implement any of these additional features or need further refinements!
