      
# Flow System: Detailed Guidelines and Conventions

This document provides detailed operational guidelines, conventions, templates, and procedures for using the Flow system effectively. It expands on the high-level principles outlined in the main `flow.md`.

## System Conventions and Configuration

This section defines standard conventions and configuration preferences used within the Flow system documentation and workflow.

### Appearance

-   **Priority Colors**:
    -   🔴 Critical - Used for urgent items that block progress
    -   🟠 High - Important items that need immediate attention
    -   🟡 Medium - Standard work items
    -   🟢 Low - Nice-to-have items or long-term improvements

-   **Status Indicators**:
    -   ⏳ In progress - Actively being worked on
    -   ✅ Completed - Work is finished and verified
    -   🔄 Under review - Awaiting feedback or assessment
    -   ⏸️ Paused - Temporarily suspended
    -   🚩 Blocked - Unable to proceed due to dependencies

### Organization

-   **Task Categories** (Examples - Adapt as needed per project):
    -   Implementation - New features or components
    -   Bug Fix - Addressing errors or defects
    -   Research - Investigation and information gathering
    -   Documentation - Improving or extending documentation
    -   Refactoring - Restructuring without changing behavior

### Default Behaviors

-   **Log Retention** (in `flow/project/logs/`):
    -   Keep recent task history in `flow/project/tasks.md`.
    -   Archive older completed task entries to the `flow/project/logs/` directory.
    -   Retention period for archived logs is project-dependent, but aim for regular archival (e.g., weekly or when `tasks.md` becomes too long). Consider date-based filenames (e.g., `logs_YYYY-MM-DD.md`).

-   **Review Frequency** (Recommended):
    -   Daily check-ins for active tasks in `project/tasks.md`.
    -   Weekly review of the entire task list and backlog (if maintained separately).
    -   Periodic consistency audits of the Flow system documentation itself.

*Note: Project-specific overrides or additions to these conventions can be documented in `flow/project/conventions.md` (if such a file is created).*

## Active Task Management Procedures

Detailed procedures for managing tasks within `../project/tasks.md`.

-   **Primary Hub:** `../project/tasks.md` is the central list for *active* tasks. Each task should include a brief description, status/priority indicators (see Conventions), and ideally a link to more detailed documentation if needed (e.g., a specific plan, issue tracker link, or design document).
-   **Task Status:** Use the defined status indicators consistently.
-   **Task Priority:** Use priority colors where appropriate to guide focus.
-   **Granularity:** Keep the `project/tasks.md` list focused on actionable, current work items. Detailed sub-tasks or long-term planning should reside in linked documents to keep the main list concise.
-   **Updates:** `project/tasks.md` should be updated continuously as work progresses. Add new tasks, update statuses, and archive completed items promptly.
-   **Timestamps & Context:** When adding tasks or significant updates, consider adding a timestamp or linking back to the originating discussion/commit for context.
-   **Archival Process:** Once a task is completed (✅) or deemed no longer active, remove it from `../project/tasks.md` and log its completion/archival in the `../project/logs/` directory, typically with a timestamp and outcome summary. Maintain only recent history (e.g., last 5-10 completed items) directly in `tasks.md` if desired for quick reference before full archival.
-   **Scheduled Reviews & Retrospectives:** Implement regular reviews (daily/weekly) to reconcile the active task list, ensure consistency, identify blockers, and capture key decisions. Consider brief retrospectives after major milestones or complex tasks to capture lessons learned.

## AI Agent Interaction Patterns

### Interpreting User Requests to "Remember"
- When the user asks the AI agent (you) to "remember" specific information, decisions, plans, or context, this should be interpreted as a directive to **record that information explicitly within the appropriate file(s) in the Flow system**.
- The goal is to persist the information beyond the immediate conversation context, ensuring it becomes part of the documented project/system state.
- Choose the most logical location (e.g., update a task in `tasks.md`, add to `glossary.md`, log a decision in `evolution.md` or a dedicated log, update `guidelines_and_conventions.md` itself, etc.). If unsure, propose a location to the user.
- This ensures that "remembering" translates into durable, accessible knowledge within Flow, aligning with the system's memory-aid goals.

## Task Templates

Use these templates as starting points for common task types in `project/tasks.md` to ensure consistency and necessary detail. Adapt as needed.

### Bug Fix Template

```markdown
- 🔴 **Fix [Bug Name]**: ⏳ [Brief description of the bug and its impact]. Link: [Link to issue tracker/detailed report]
  - Steps: Identify cause, Implement fix, Test, Update docs.

- 🟡 **Implement [Feature Name]**: ⏳ [Brief feature description]. Link: [Link to spec/design doc]
  - Dependencies: [Prerequisite tasks/features]
  - Acceptance Criteria: [How to verify completion]

- 🟢 **Investigate [Topic]**: ⏳ [Goal of the investigation]. Link: [Link to research notes/document]
  - Key Questions: [Question 1?], [Question 2?]
  - Outcome: [Expected deliverable, e.g., summary, recommendation]

## Minimizing Cruft and Managing Temporary Files

To prevent the accumulation of unused or temporary files ("cruft"), especially when using automated tools or AI agents:

    Dedicated Scratch/Experiment Directories:

        Use a top-level .gitignore-ed directory like scratch/ or temp/ for temporary scripts, outputs, data samples, and exploratory work. Files here are considered ephemeral.

    Permanent Output Location:

        Ensure permanent artifacts (builds, final reports, datasets) are placed in designated project directories (e.g., dist/, artifacts/, data/processed/). Do not leave permanent outputs in temporary directories.

    Agent Output Containment:

        Instruct AI agents to place newly generated files within a designated subdirectory (e.g., scratch/agent_output/) unless explicitly intended as permanent additions to core directories (src/, tests/, docs/).

    Regular Cleanup:

        Periodically review and clean out temporary directories.

        Consider adding a "Cleanup temporary files" step to relevant task checklists in project/tasks.md.

    Explicit Instructions:

        Clearly communicate expectations regarding file placement and lifespan when initiating tasks involving file creation, especially with AI agents.

## Maintaining Documentation Consistency

Strategies to prevent inconsistencies within the Flow documentation system itself:

*Refer to the Core Principles defined in `flow.md` for operational guidelines.*

### Consistency Maintenance Practices

    Regular Audits: Periodically review Flow documents for consistency, broken links, and duplicated information.

    Change Management: Update authoritative sources first (like this file), then propagate changes or update references in other documents. Document significant changes (perhaps in `../meta/evolution.md` or a dedicated `../meta/CHANGELOG.md`).

    Refactoring Sessions: Occasionally review the Flow system structure itself for potential improvements or simplification. Remove outdated documents or sections.