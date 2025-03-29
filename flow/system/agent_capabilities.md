# AI Agent Capabilities

This document outlines key capabilities and limitations of the AI agent assisting with development.

## File System Interaction

*   The agent **can** interact with the workspace file system using provided tools.
*   This includes:
    *   Reading specific file contents or sections (`read_file`).
    *   Listing directory contents (`list_dir`).
    *   Searching for files by name (`file_search`).
    *   Searching file contents using regex (`grep_search`).
    *   Proposing edits to existing files (`edit_file`).
    *   Creating new files (`create_file`).
    *   Deleting files (`delete_file`).
*   These actions require specific instructions, including target file paths or search queries.
*   The agent **cannot** autonomously browse the file system or monitor files/processes in real-time without explicit tool calls.

## Other Capabilities

*   Executing terminal commands (requires user approval) (`run_terminal_cmd`).
*   Performing web searches (`web_search`).
*   Performing semantic code searches (`codebase_search`).

*Note: Capabilities are subject to change based on the available tools and system configuration.* 