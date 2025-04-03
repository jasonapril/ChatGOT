# Meta Tasks

This document tracks tasks related to the development and improvement of the Flow system itself.

## Active Tasks

### Strengthen Flow Indexing & Entry Points 🟡 ✅
- **Description**: Improve README.md files in `flow/`, `system/`, `project/`, `meta/` to serve as comprehensive indexes, clearly listing key files and their purposes. Consider a master `flow/index.md`.
- **Notes**: Aims to improve AI discoverability of relevant context without explicit user prompting. Decided against `index.md` as improved READMEs suffice.
- **Status**: Completed (2025-03-27)
- **Links**: [README](../README.md), [Principles](principles.md)

### Define AI Interaction Protocols for Flow 🟡 ✅
- **Description**: Document the expected process for how an AI agent should use Flow to retrieve information contextually and efficiently (e.g., read READMEs, check tasks.md, use search).
- **Notes**: Could go in `system/guidelines.md` or a new `system/ai_interaction_protocols.md`. Should specify retrieval priorities. Added section to `system/guidelines.md`.
- **Status**: Completed (2025-03-27)
- **Links**: [Guidelines](../system/guidelines.md), [Principles](principles.md)

### Review & Consolidate Flow Documentation 🟡 ✅
- **Description**: Evaluate `flow/` subdirectories, particularly `meta/`, for potential consolidation of related files (e.g., merging multiple troubleshooting guides).
- **Notes**: Goal is to reduce the number of discrete files an AI needs to search/parse, improving discoverability. Balance consolidation with clarity. Merged troubleshooting/debugging files; removed reminders.md and process_notes_from_state.md.
- **Status**: Completed (2025-03-27)
- **Links**: [Evolution](evolution.md), [Principles](principles.md)

### Implement Priority Continuity Mechanisms 🔴
- **Description**: Create a system to maintain priority continuity across planning cycles, ensuring important tasks don't get lost during transitions
- **Notes**: Should include documentation on how to carry priorities forward and adjust as needed
- **Status**: Planning
- **Links**: [Evolution](evolution.md)

### Localize Domain Knowledge for Craft Project 🟠
- **Description**: Document and organize domain-specific knowledge for the Craft project
- **Notes**: Initial structure created in reference/craft_domain.md; needs detailed expansion
- **Status**: Started
- **Links**: [Domain Knowledge Outline](../reference/craft_domain.md)

### Develop Flow Glossary 🟡
- **Description**: Create a comprehensive glossary of Flow terminology and concepts
- **Notes**: Will help clarify distinctions between terms like "project," "plan," and "goal"
- **Status**: Not started
- **Links**: [Model](model.md)

### Implement Prospective Memory Features 🟡
- **Description**: Develop mechanisms for remembering to perform planned actions at appropriate times
- **Notes**: Could include scheduled reviews, reminders, and temporal markers
- **Status**: Conceptual
- **Links**: [Model](model.md)

## Upcoming Tasks

### Make System for Goals & Ensure Tasks Align with Goals Even If the Goals Change

### Create Short-term Buffer Mechanism
- **Description**: Implement a system for temporarily storing ideas and thoughts before proper organization
- **Notes**: Should be lightweight and require minimal effort to use
- **Links**: [Model](model.md)

### Enhance Focus Mechanisms
- **Description**: Develop better tools for directing attention to high-priority tasks
- **Notes**: Consider implementing a "current focus" section in active tasks
- **Links**: [Model](model.md)

### Develop Meta Project Structure
- **Description**: Clarify how meta documentation relates to the rest of Flow
- **Notes**: Need to determine if meta should be treated as a project, plan, or something else
- **Links**: [Evolution](evolution.md)

### Implement Sidetrack Detection
- **Description**: Create a mechanism for Flow to detect when work is getting too far from priority tasks
- **Notes**: Could compare current work against prioritized tasks to identify divergence
- **Links**: [Model](model.md)

### Define Domain Knowledge Storage
- **Description**: Decide on and document a standard location and format for storing general domain knowledge (not specific to Flow or the current project).
- **Notes**: Should this be in `docs/` or a dedicated top-level `knowledge/` directory? How should it be organized?
- **Status**: Not started
- **Links**: (Potentially link to relevant Flow guidelines)

### Review Project Task Structure
- **Description**: Evaluate the structure of project-specific task lists (e.g., `flow_craft/active/tasks.md`).
- **Notes**: Consider if separating upcoming/backlog tasks from active tasks improves clarity. Update `flow/system/guidelines.md` with recommendations.
- **Status**: Not started
- **Links**: [Guidelines](../system/guidelines.md)

## User-Added Tasks to Review

### Technical Solutions for Consistency

    Linters/Validators: Consider simple scripts to check for valid internal links or consistent use of status markers.

    Templates: Use and maintain standardized templates (like those above) for recurring elements.

### Automation Considerations

To streamline Flow system usage:

    Lightweight Scripting: Consider simple scripts or Git hooks to automate parts of the task update/archival process (e.g., a script to move ✅ tasks from project/tasks.md to project/logs/).

    Automated Reminders: Set up reminders for periodic reviews or consistency checks.

    Minimize Manual Work: The goal of automation should be to reduce repetitive manual effort in maintaining the Flow system, allowing more focus on the actual project tasks.


### Simplify Flow Goals
- Flow gives AI agents memory and executive functioning.

### Generalize Project Details Mentioned Outside the Project
- Standardize top-level files and folders
- Consider templates for different types of projects (see domains)

### Add Scripts to Keep Flow's Data Standardized, Organized, Updated, and Integrous
- Functional Alignment: Ensure that all tasks/plans/projects/roadmaps are aligned with stated goals/principles.
- Ensure that task dependencies are updated.
- Ensure that architectural decisions are upheld.
- Move items to backlog or associated projects/plans/roadmaps to clear space in memory.

### Review Terminology
- Should tasks have subtasks?
- What's the difference between plans, projects, roadmaps, goals, etc.?

### Standardize Tasks to Always Begin with Verbs?

### Identify and Decide How to Handle Errors, Debugging, and Troubleshooting

### Understand How to Remember

- We need to define exactly how you "remember": by storing information in files like this one. But we should be careful that the information is effectively organized.

### Learn from Mistakes
- You (the AI agent) need to recognize when you're in troubleshooting mode as a result of your own error, solve the problem, then perform a sort of post mortem to better understand what you could have done to prevent the problem or solve the problem more effectively, then generalize that understanding (as much as possible) to your instructions to incorporate the newfound knowledge in the future.
- In short, you should be able to recognize your mistakes and learn from them by "remembering."

### Rename `active` Folder to `current`?
- Maintains a theme. A current flows. Too much?

### Implement Time Management Feature
- This may require scripts more than documentation.
- Closely related to Implement Prospective Memory Features task

### Find Way to Get Time of Day
- This is critical for time management and logs. Timestamps are generally useful properties.

## Completed Tasks

### Strengthen Flow Indexing & Entry Points
- **Description**: Improved README files in `flow/`, `system/`, `project/`, `meta/` to serve as better indexes.
- **Completion Date**: 2025-03-27
- **Outcome**: Enhanced `flow/README.md`, created `project/README.md`, updated `meta/README.md`.

### Create Meta Directory Structure
- **Description**: Establish a dedicated directory for Flow's self-documentation
- **Completion Date**: 2025-03-26
- **Outcome**: Created meta directory with initial files for model, evolution, and tasks

### Document Memory Model
- **Description**: Create comprehensive documentation of Flow's memory model
- **Completion Date**: 2025-03-26
- **Outcome**: Created model.md documenting memory systems and cognitive enhancement strategies

### Document Flow Evolution
- **Description**: Create documentation tracking Flow's development history and trajectory
- **Completion Date**: 2025-03-26
- **Outcome**: Created evolution.md capturing phases of development and future directions

### Define AI Interaction Protocols for Flow
- **Description**: Documented the expected process for AI information retrieval from Flow.
- **Completion Date**: 2025-03-27
- **Outcome**: Added "AI Agent Interaction with Flow" section to `flow/system/guidelines.md`.

### Review & Consolidate Flow Documentation
- **Description**: Reviewed `flow/meta/` for consolidation opportunities to improve discoverability.
- **Completion Date**: 2025-03-27
- **Outcome**: Merged `debugging_strategy.md`, `general_troubleshooting_framework.md`, `post_mortem_rule.md` into `troubleshooting_and_debugging.md`. Integrated relevant content from and deleted `reminders.md` and `process_notes_from_state.md`. Updated `meta/README.md` index.

## Task Log

- **2025-03-27**: Completed task "Review & Consolidate Flow Documentation".
- **2025-03-27**: Completed task "Define AI Interaction Protocols for Flow" by adding section to guidelines.
- **2025-03-27**: Completed task "Strengthen Flow Indexing & Entry Points" by enhancing README files.
- **2025-03-27**: Created meta directory and initial documentation 