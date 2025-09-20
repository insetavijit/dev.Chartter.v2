#!/usr/bin/env node

const fs = require('fs').promises; // Use promises for async operations
const path = require('path');
const yargs = require('yargs'); // For command-line argument parsing

// Parse command-line arguments
const argv = yargs
    .option('ext', {
        alias: 'e',
        type: 'string',
        default: 'md',
        description: 'File extension for the note (e.g., md, txt)',
    })
    .option('template', {
        alias: 't',
        type: 'string',
        description: 'Path to a custom template file',
    })
    .option('name', {
        alias: 'n',
        type: 'string',
        description: 'Custom name for the note file (overrides datetime)',
    })
    .help()
    .argv;

// Get current date and datetime
const now = new Date();
const dateFolder = now.toISOString().split('T')[0]; // YYYY-MM-DD
const datetimeFile = now.toISOString().replace(/[:.]/g, '-'); // YYYY-MM-DDTHH-MM-SS-sssZ

// Determine file name
const fileName = argv.name ? `${argv.name}.${argv.ext}` : `${datetimeFile}.${argv.ext}`;

// Paths
// Base folder: Notes inside current working directory
const notesFolder = path.join(process.cwd(), "Notes");
const folderPath = path.join(notesFolder, dateFolder);
const filePath = path.join(folderPath, fileName);

// Default template if none provided
const defaultTemplate = `# Notes for ${fileName.replace(`.${argv.ext}`, '')}\n\n`;

// Function to read template file
async function getTemplateContent() {
    if (argv.template) {
        try {
            return await fs.readFile(argv.template, 'utf8');
        } catch (err) {
            console.error(`❌ Error reading template file: ${err.message}`);
            process.exit(1);
        }
    }
    return defaultTemplate;
}

// Main function to create the note
async function createNote() {
    try {
        // Ensure folder exists
        await fs.mkdir(folderPath, { recursive: true });

        // Get template content
        const content = await getTemplateContent();

        // Create file with content
        await fs.writeFile(filePath, content);

        // Log success with relative path for better readability
        const relativePath = path.relative(process.cwd(), filePath);
        console.log(`✅ Created file: ${relativePath}`);
    } catch (err) {
        console.error(`❌ Error creating note: ${err.message}`);
        process.exit(1);
    }
}

// Run the script
createNote();
