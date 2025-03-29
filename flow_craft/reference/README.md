# Reference Directory

This directory contains domain-specific knowledge and technical implementation details that serve as Flow's semantic memory. Unlike procedural memory (how to do things) in the `system/` directory or working memory (current tasks) in the `active/` directory, reference materials focus on factual information and technical details needed for project work.

## Contents

- [craft_domain.md](craft_domain.md) - Domain knowledge outline for the Craft project
- [config_implementation.md](config_implementation.md) - Technical details about configuration implementation

## Purpose

The reference directory serves several key functions:

1. **Knowledge Preservation** - Documenting technical details that might otherwise be forgotten
2. **Onboarding Support** - Providing domain context for anyone new to the project
3. **Decision Reference** - Recording why specific technical approaches were chosen
4. **Implementation Guidance** - Offering detailed specifications for development work

## Organization

Reference materials are organized by:

1. **Project** - Each major project has its own reference document(s)
2. **Domain Area** - Within projects, information is organized by technical domain
3. **Specificity** - From general concepts to specific implementation details

## Usage Guidelines

- Reference materials should be technical and specific
- Focus on "what" and "why" rather than "how" (procedures belong in `system/`)
- Include code examples, diagrams, and technical specifications where helpful
- Link to external resources for background information when appropriate
- Update reference materials when technical implementations change

## Related Resources

- [System Guidelines](../system/guidelines.md) - Procedures and workflows
- [Meta Model](../meta/model.md) - How reference fits into Flow's memory model
- [Domain Knowledge Task](../meta/tasks.md) - Current task for expanding domain knowledge 