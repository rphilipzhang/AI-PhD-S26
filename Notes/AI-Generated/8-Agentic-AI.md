# Agentic AI

**DOTE 6635: Artificial Intelligence for Business Research (Spring 2026)**

**Instructor: Renyu (Philip) Zhang**

## Abstract

This article provides a comprehensive introduction to Agentic AI—the emerging paradigm in which large language models (LLMs) move beyond generating text to autonomously *acting* in the world. The content is based on the lecture slides from the course "DOTE 6635: Artificial Intelligence for Business Research" and is supplemented with additional explanations and references to foundational literature. We begin by defining what constitutes an AI agent and tracing the conceptual evolution from conversational chatbots to autonomous systems capable of planning, tool use, and multi-step reasoning. We then examine the infrastructure that enables agents—**skills** as standard operating procedures (SOPs), including the three primary skill archetypes, the anatomy of a skill, the progressive disclosure architecture for token efficiency, frontmatter-based triggering mechanisms, and the principles of nonambiguous instruction design—and the **[Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol)** as a universal standard for connecting agents to external tools, data, and applications. Next, we conduct a detailed case study of **[OpenClaw](https://openclaw.ai/)**, an open-source agentic system whose viral adoption in early 2026 epitomizes both the promise and peril of consumer-facing AI agents. We analyze its architecture—unified context, self-evolving memory, and the ecosystem flywheel—before discussing security vulnerabilities (the "lethal trifecta") and a principled reimplementation strategy. Finally, we turn to **[context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)**, the discipline of curating and managing the optimal set of information delivered to an LLM at inference time, covering system prompt design, token-efficient tool definitions, progressive information retrieval, compaction strategies for long-horizon tasks, structured note-taking for agentic memory, and sub-agent architectures. Throughout, we emphasize the shift from *prompt engineering* (crafting a single instruction) to *context engineering* (managing an evolving information state across an extended agentic loop)—a shift that defines the frontier of applied AI research.

## 1. Where Are We?

Before diving into agentic AI, it is helpful to situate this lecture within the broader arc of the course:

- **Reasoning LLMs** serve as the backbone for AI agents. Models capable of chain-of-thought (CoT) reasoning, planning, and self-correction provide the "brain" that powers autonomous behavior.
- **LLM-driven research automation**—replication automation, Automated Prompt Engineering (APE), Autoresearch, and FARS—has already demonstrated that LLM agents can drive paradigm shifts in academic research workflows.
- **Agent-based modeling** (e.g., structural models in economics and operations) has been an important and long-standing methodology in business, economics, and social science. AI agents extend this tradition with learned, adaptive policies rather than hand-crafted behavioral rules.
- **AI agents are impacting real business** today, from customer service chatbots and market research copilots to coding assistants and supply chain robotics.

A useful way to think about progress is the **automation ladder**:

1. **Robustness:** Automating tedious digital labor (e.g., data entry, report generation).
2. **Collaboration:** Facilitating human interactions (e.g., scheduling, email drafting, meeting summarization).
3. **Exploration:** Enabling creativity and scientific discovery (e.g., hypothesis generation, experimental design).

> **The fundamental transition:** From ChatGPT to OpenClaw—from *AI that talks* to *AI that acts*.

## 2. LLM Agents

### 2.1. What Is an Agent?

An **agent** is an intelligent system that interacts with some environment via a cycle of actions and observations (see the [Language Agent Tutorial](https://language-agent-tutorial.github.io/slides/I-Introduction.pdf) [1] and the [LLM Agents MOOC](https://llmagents-learning.org/slides/intro.pdf) [2] for comprehensive introductions). This definition is deliberately broad: it encompasses classical AI agents (rule-based systems, search algorithms), neural agents (learned policies via reinforcement learning), and the new generation of **language agents** that use natural language as the vehicle for both reasoning and communication.

What distinguishes modern LLM agents from earlier generations is the *generality* of the language interface. Classical agents typically required domain-specific state representations and hand-crafted action primitives. LLM agents, by contrast, can:

- **Reason** in natural language about complex, open-ended tasks.
- **Communicate** with humans and other agents using the same natural language.
- **Generalize** across domains without task-specific retraining, leveraging the broad world knowledge embedded in pretrained models.

### 2.2. From LLM to Agents

A standalone LLM can be thought of as a *brain in a vat*—immensely knowledgeable but unable to perceive or act upon the world. The transition from an LLM to an agent involves equipping this brain with **interactions with the environment**: the ability to observe states, take actions, receive feedback, and iteratively refine its approach ([Language Agent Tutorial](https://language-agent-tutorial.github.io/slides/I-Introduction.pdf) [1]; [LLM Agents MOOC](https://llmagents-learning.org/slides/intro.pdf) [2]).

This transition requires several capabilities beyond text generation:

- **Perception:** Processing inputs from the environment (user messages, tool outputs, error logs, sensor readings).
- **Reasoning and Planning:** Decomposing high-level goals into ordered sub-tasks, selecting among strategies, and revising plans in light of new information.
- **Action:** Executing steps via tool calls, API invocations, code execution, or message generation.
- **Memory:** Maintaining both short-term context (within the conversation) and long-term knowledge (persisted across sessions).

### 2.3. Conceptual Framework of LLM Agents

The conceptual framework for LLM agents, as articulated in recent tutorials ([Language Agent Tutorial](https://language-agent-tutorial.github.io/slides/I-Introduction.pdf) [1]; [LLM Agents MOOC](https://llmagents-learning.org/slides/intro.pdf) [2]), revolves around the **agentic loop**—the iterative cycle through which agents solve real-world tasks:

```
┌──────────────────────────────────────────────────┐
│                   Agentic Loop                    │
│                                                   │
│   Observation ─→ Reasoning ─→ Action ─→ Feedback  │
│        ↑                                    │     │
│        └────────────────────────────────────┘     │
└──────────────────────────────────────────────────┘
```

**Key properties of the agentic loop:**

1. **Trial-and-error.** Solving real-world tasks typically involves iterative refinement. The agent attempts an action, observes the result, and adjusts its approach—much like how a human programmer debugs code by running it, reading error messages, and modifying the source.

2. **External tool use.** Leveraging external tools (calculators, search engines, APIs, databases) and retrieving from external knowledge bases expand the LLM's capabilities beyond its parametric knowledge. Tool use transforms the model from a knowledge retrieval system into an *action-taking* system.

3. **Task decomposition and subtask allocation.** Agentic workflows facilitate complex tasks through decomposition into manageable sub-problems, allocation and parallelization of subtasks to specialized modules, and division of labor for project collaboration.

4. **Multi-agent generation.** Having multiple agents collaborate—each with distinct roles, expertise, or perspectives—can inspire better responses through debate, verification, and synthesis, analogous to ensemble methods in machine learning.

### 2.4. The Agent as a Smart PhD with SOPs

A useful metaphor, articulated by Hung-yi Lee in his [Machine Learning course](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/intro.pdf) [3], frames LLM agents as **smart, knowledgeable PhDs who need clear standard operating procedures (SOPs)**. Just as a newly hired researcher—no matter how brilliant—needs onboarding documents, lab protocols, and institutional workflows to be productive, an LLM agent needs structured instructions (skills), access to tools (MCPs), and well-organized context to perform effectively.

This metaphor highlights an important asymmetry: the bottleneck is often not the agent's *intelligence* but the *quality of its operating instructions and environment*. Improving agent performance is frequently more about better context engineering than better models.

## 3. Skills and the Model Context Protocol (MCP)

### 3.1. Agent Skills

**Skills** are, fundamentally, SOPs for LLM agents to complete specific tasks (see [Lee, 2026](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/intro.pdf) [3]; [Anthropic's Guide to Building Skills](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) [4]; [DeepLearning.AI Short Course](https://www.deeplearning.ai/short-courses/agent-skills-with-anthropic/) [5]). Recall the metaphor from Section 2.4: LLM agents are smart, knowledgeable PhDs who need clear SOPs. Skills formalize these SOPs into modular, reusable components that encode domain-specific expertise.

A skill encapsulates:

- **Instructions:** A structured prompt defining what the agent should do, how it should do it, and what constraints apply.
- **Metadata:** Information about when the skill should be triggered, what tools it requires, and what outputs it produces.
- **Best practices:** Curated patterns from expert users or the developer community that encode domain-specific knowledge.

Skills can be:
- **Authored by developers** who understand a particular workflow.
- **Downloaded from repositories** such as the [Anthropic Skills repository](https://github.com/anthropics/skills) [4] or community hubs like [AgentSkills.io](https://agentskills.io/home) [15].
- **Generated by AI agents themselves**—an agent can write new skills based on observed best practices, creating a self-evolution loop (more on this in Section 4).

The skill paradigm represents a shift from monolithic prompt engineering to **modular, reusable, and composable** agent capabilities.

### 3.2. Skill Archetypes

[Anthropic's guide](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) [4] identifies **three primary skill archetypes**, each suited to a different class of tasks:

| Archetype | Purpose | External Tools? | Example |
|-----------|---------|:---------------:|---------|
| **Document & Asset Creation** | Creates consistent, production-grade outputs (presentations, code, designs, documents) | No — leverages the LLM's built-in capabilities | A "frontend-design" skill that generates distinctive UI components following a design system |
| **Workflow Automation** | Manages multi-step processes with templates, validation gates, and iterative refinement loops | Optional | A "skill-creator" skill that walks the agent through use-case definition, frontmatter generation, instruction writing, and validation |
| **MCP Enhancement** | Layers domain expertise on top of MCP tool access, turning raw API connectivity into reliable guided workflows | Yes — depends on MCP servers | A code-review skill that analyzes GitHub PRs using Sentry error data via its MCP server |

The key distinction: **Document & Asset Creation** skills are self-contained (no external tools needed), **Workflow Automation** skills are process-oriented with validation loops, and **MCP Enhancement** skills coordinate external tool calls with expert guidance. In practice, many production skills blend elements from multiple archetypes.

### 3.3. Anatomy of a Skill

While the file architecture may differ across agent platforms, the underlying logic remains the same ([Anthropic's Guide](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) [4]; [DeepLearning.AI Short Course](https://www.deeplearning.ai/short-courses/agent-skills-with-anthropic/) [5]). A typical skill directory follows this structure:

```
skills/
└── skill-name/
    ├── SKILL.md          # Main skill definition (required)
    ├── references/       # Detailed documentation and patterns
    │   └── patterns.md
    ├── examples/         # Complete, runnable examples
    │   └── sample.md
    └── scripts/          # Executable utilities and validators
        └── helper.sh
```

- **`SKILL.md`** is the required entry point—it contains YAML frontmatter (metadata) and the markdown body (instructions).
- **`references/`** holds detailed documentation, advanced techniques, and edge-case guidance that the agent loads only when needed.
- **`examples/`** contains complete, runnable examples and templates.
- **`scripts/`** houses executable utilities whose *output* (not source code) enters the context window.

This modular structure is critical for context efficiency: the agent loads only what it needs, when it needs it—a principle formalized as *progressive disclosure* (Section 3.4).

### 3.4. Progressive Disclosure

A core architectural insight behind skills is **progressive disclosure**: the principle that LLM agents should access deep knowledge without easily exhausting context windows ([Anthropic's Guide](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) [4]; [DeepLearning.AI Short Course](https://www.deeplearning.ai/short-courses/agent-skills-with-anthropic/) [5]). Progressive disclosure minimizes token usage while maintaining access to specialized expertise through a three-level loading system:

| Level | What Is Loaded | When | Approximate Token Cost |
|-------|----------------|------|:----------------------:|
| **1. Metadata** | `name` and `description` from YAML frontmatter | Always (at startup, injected into system prompt) | ~50–100 tokens per skill |
| **2. Instructions** | Full `SKILL.md` body | When the skill is triggered | Up to ~5,000 tokens |
| **3. Resources & Code** | Files in `references/`, `examples/`, `scripts/` | On-demand, only when referenced by instructions | Effectively unlimited |

**How it works:** At startup, the agent loads *only* Level 1 metadata from every installed skill into its system prompt. This costs negligible tokens but gives the agent enough context to recognize when a skill is relevant. When triggered, the full `SKILL.md` body (Level 2) is read into context. Detailed reference materials and executable scripts (Level 3) are loaded only when the instructions explicitly reference them—and for scripts, only the *output* enters the context window, not the source code itself.

The analogy is a well-organized manual: the table of contents (Level 1) is always visible, specific chapters (Level 2) are opened as needed, and the detailed appendix (Level 3) is consulted only for specialized questions. This layered approach enables dozens of skills to coexist with minimal context overhead—the agent pays the full token cost only for the skill it is actively using.

### 3.5. Frontmatter: When Skills Are Triggered

The **frontmatter** is the YAML metadata block at the top of `SKILL.md`. It is loaded into the system prompt and serves as the primary mechanism for determining *when* a skill is activated ([Anthropic's Guide](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) [4]):

```yaml
---
name: skill-name
description: This skill should be used when the user asks to "specific phrase 1",
  "specific phrase 2", or mentions specific-topic. Provides [capability].
---
```

**Triggering mechanism:** When a user makes a request, the agent evaluates it against all available skill descriptions using semantic matching. If the request aligns with a skill's description, the agent issues a skill invocation, which loads the full `SKILL.md` body into context.

**Best practices for the `description` field:**

- **Write in third person** — the description is injected into the system prompt alongside other metadata, so first-person ("I can help you...") or second-person ("You can use this to...") creates point-of-view inconsistency.
- **Include specific trigger phrases** that users would naturally say, enclosed in quotes (e.g., `"create a PR"`, `"review this code"`).
- **Be concrete about activation conditions** — state both *what* the skill does and *when* it should be used.
- **Be assertive** — include language like "Make sure to use this skill whenever..." to encourage reliable triggering.

The description is the single most critical field in the entire skill definition. A vague description (e.g., "Helps with PDFs") will fail to trigger reliably; a specific one (e.g., "This skill should be used when the user asks to 'rotate a PDF', 'merge PDFs', or mentions PDF manipulation") will trigger consistently.

### 3.6. Nonambiguous Instructions

A fundamental challenge in skill design is that **code is deterministic, but natural language is not** ([Anthropic's Guide](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) [4]; [DeepLearning.AI Short Course](https://www.deeplearning.ai/short-courses/agent-skills-with-anthropic/) [5]). The same instruction may be interpreted differently across invocations due to the probabilistic nature of LLMs. Effective skills write instructions that leave no room for guesswork:

**1. Replace vague directives with explicit sequences.**
- Vague: "Validate the data."
- Explicit: "Before calling `create_project`, verify: (a) project name is non-empty, (b) at least one team member is assigned, (c) start date is not in the past."

**2. Use code for deterministic operations.** For critical validations, bundle a script rather than relying on language instructions. Claude handles qualitative reasoning (understanding user intent) while scripts provide quantitative precision (deterministic execution). Only the script's *output* enters the context window.

**3. Calibrate the degree of freedom.** Not all instructions require the same precision. The appropriate level of specificity depends on the task:

| Degree of Freedom | When to Use | Example |
|-------------------|-------------|---------|
| **High** (text instructions) | Multiple valid approaches; decisions depend on context | Code review, creative writing |
| **Medium** (pseudocode / parameterized scripts) | A preferred pattern exists but some variation is acceptable | API integration workflows |
| **Low** (specific scripts, exact sequences) | Operations are fragile or consistency is critical | Database migrations, deployment pipelines |

**4. Structure instructions clearly.** Effective skill instructions follow a consistent format: a brief purpose statement, prerequisites, step-by-step procedures in imperative mood ("Analyze...", "Execute..."), output format definitions, and error handling guidance with common failure scenarios.

You can download useful skills from the [Anthropic Skills repository](https://github.com/anthropics/skills) [4] or community hubs like [AgentSkills.io](https://agentskills.io/home) [15], and ask your AI agents to write skills for you based on these best practices.

### 3.7. The Model Context Protocol (MCP)

The **[Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol)** is an open standard introduced by Anthropic in November 2024 for connecting AI agents to external applications, data sources, and tools [6]. It provides a universal, standardized interface—analogous to how the Language Server Protocol (LSP) standardized how programming language support works across development tools. The full protocol specification is available at [modelcontextprotocol.io](https://modelcontextprotocol.io/) [6].

**The problem MCP solves:** Before MCP, every new data source or tool required a bespoke integration. With $M$ AI applications and $N$ tools, this created an $M \times N$ integration problem. MCP replaces this combinatorial explosion with a single standard: each application implements one MCP client, and each tool implements one MCP server, reducing the problem to $M + N$.

**Architecture:** MCP follows a **client-host-server** model [6]:

| Component | Role | Example |
|-----------|------|---------|
| **Host** | The top-level AI application that the user interacts with. Coordinates one or more MCP clients. | Claude Desktop, Claude Code, Cursor, an IDE |
| **Client** | A connector within the host that maintains a dedicated 1:1 connection to a single MCP server. | One client per connected service |
| **Server** | A lightweight program exposing specific capabilities (tools, data, prompts) through the MCP protocol. | A GitHub server, a database server, a calendar server |

A single host can connect to many servers simultaneously—e.g., one for file system access, one for a database, one for GitHub, one for Slack—each through its own dedicated client instance.

**Communication:** MCP uses JSON-RPC 2.0 messages over two transport options:
- **stdio:** Local process communication with no network overhead—ideal for tools running on the same machine.
- **Streamable HTTP:** Remote servers accessible over the network, with optional Server-Sent Events for streaming responses.

**Core primitives exposed by MCP servers:**

1. **Tools:** Executable functions the AI can invoke (e.g., file operations, API calls, database queries). Discovered via `tools/list`, executed via `tools/call`.
2. **Resources:** Data sources providing contextual information (e.g., file contents, database records, calendar entries).
3. **Prompts:** Reusable templates for structuring LLM interactions (e.g., domain-specific system prompts).

**Industry adoption:** MCP has achieved rapid adoption. OpenAI adopted MCP in March 2025, Google DeepMind confirmed support in April 2025, and Microsoft followed shortly after. Development tool companies (Zed, Replit, Codeium, Sourcegraph) and enterprises have integrated MCP into production agent systems [6].

> **Why MCP matters for researchers:** MCP lowers the barrier to building agents that interact with real-world systems. A researcher can now connect an LLM agent to their institutional database, their email, their calendar, and their codebase through a single protocol—without writing custom integrations for each.

## 4. OpenClaw: A Case Study in Agentic AI

### 4.1. What Is OpenClaw?

**[OpenClaw](https://openclaw.ai/)** (formerly Clawdbot, then MoltBot) is a free and open-source agentic system that executes tasks via LLMs using messaging platforms (WhatsApp, Lark, Telegram, Discord, Signal) as its primary user interface (see the [Yage.ai deep dive](https://yage.ai/openclaw-en.html) [7]; [OpenClaw GitHub](https://github.com/openclaw/openclaw) [8]; [NVIDIA GTC keynote](https://www.nvidia.com/gtc/keynote/) [9]). Created by Austrian developer [Peter Steinberger](https://steipete.me/posts/2026/openclaw) [14], it was first published in November 2025 and went viral in late January 2026.

OpenClaw represents a paradigm shift: instead of interacting with AI through a specialized interface (a chat window, an IDE), users interact through the messaging apps they already use daily. The agent runs locally on the user's machine, connecting to LLM providers (Claude, DeepSeek, GPT) and executing tasks autonomously.

**Key statistics at peak virality (February 2026):**
- Over 145,000 GitHub stars (a record for an open-source project).
- 2 million visitors per week at peak traffic.
- Over 30,000 publicly exposed instances.

### 4.2. Behind OpenClaw's Viral Growth

OpenClaw's viral success mirrors the dynamics of DeepSeek: **democratizing capabilities previously limited to developers**. By connecting agentic AI to everyday messaging apps, OpenClaw gave non-technical users their first real taste of autonomous AI agents. The key insight was that *approachability matters more than raw capability* for mass adoption.

The platform's growth was amplified by mainstream visibility—[NVIDIA's GTC 2026 keynote](https://www.nvidia.com/gtc/keynote/) [9] highlighted OpenClaw as an exemplar of the emerging **token economy** in agentic AI. The speculative frenzy around the platform reached absurd heights: during a brief window when the original "Clawdbot" social media handle was released (due to a trademark dispute with Anthropic), cryptocurrency scammers launched the fraudulent $CLAWD token, which reached a \$16 million market cap before collapsing. OpenClaw itself has no token or blockchain component, but the incident illustrates the volatile intersection of agentic AI hype and speculative finance ([OpenClaw official site](https://openclaw.ai/) [8]).

### 4.3. The Token Economy and Alibaba Token Hub

OpenClaw's emergence coincided with—and was amplified by—a broader paradigm shift that Jensen Huang articulated as the **token economy** during [NVIDIA's GTC 2026 keynote](https://www.nvidia.com/gtc/keynote/) [9]. The core thesis is deceptively simple: **tokens are the new commodity of the AI era**. Just as the industrial revolution produced physical goods and the information age produced software, the AI age produces tokens—the fundamental unit of AI output (text, code, images, reasoning steps, agent actions). Huang used the word "token" more than 70 times in a nearly two-hour speech and introduced a provocative formula for the new economics:

$$\text{Revenue} = \text{Tokens per Watt} \times \text{Available Gigawatts}$$

In this framing, data centers are no longer generic compute facilities—they are **"token factories"** whose primary output is tokens, just as power plants produce electricity. NVIDIA GPUs are the "generators" in these factories. Huang declared: *"The future data center is a token factory... Inference is your workloads and tokens are your new commodity. We have reached that moment—inference inflection has arrived"* [9]. He further argued that agentic AI would cause a massive explosion in token demand—potentially 100x compared to simple query-response AI—because each agent action (planning, tool calling, reasoning, self-correction) generates and consumes tokens across extended loops.

**Alibaba Token Hub (ATH)** provides a striking real-world illustration of this vision. On March 16, 2026—the same week as Huang's GTC keynote—Alibaba Group established ATH as a new **top-level Business Group** directly led by CEO Eddie Wu (吴泳铭), elevating it to first-tier status alongside Alibaba Cloud and e-commerce ([SCMP, 2026](https://www.scmp.com/tech/big-tech/article/3346789/alibaba-reshuffles-ai-units-new-token-hub-group-led-ceo-eddie-wu) [16]; [Alizila, 2026](https://www.alizila.com/alibaba-establishes-alibaba-token-hub-business-group/) [17]; [TechNode, 2026](https://technode.com/2026/03/17/alibaba-group-forms-alibaba-token-hub-unit-ceo-eddie-wu-to-lead-ai-push/) [18]). ATH consolidates five previously separate AI units into a single organizational structure:

| Unit | Role |
|------|------|
| **Tongyi Laboratory** | Develops Qwen foundation models (the "power plant" producing tokens) |
| **MaaS (Model-as-a-Service)** | Builds the inference infrastructure platform (the "transmission network") |
| **Qwen Business Unit** | Consumer-facing personal AI assistant (100M+ MAU) |
| **Wukong Business Unit** | B2B agentic work platform for enterprise workflows (newly created) |
| **AI Innovation Business Unit** | Explores emerging agentic consumer services (food ordering, travel, payments) |

Eddie Wu's mission statement for ATH is crystalline: **"Create tokens, deliver tokens, and apply tokens."** He framed the reorganization using an electrical grid analogy—Tongyi Lab is the power plant (producing tokens/intelligence), the MaaS platform is the transmission network (distributing tokens), and consumer/enterprise products (Qwen, Wukong) are the appliances consuming tokens. In his internal letter, Wu wrote: *"We are standing at the threshold of an AGI inflection point. Billions of AI agents are poised to take on an ever-greater share of digital work, each powered by tokens generated by models"* ([SCMP, 2026](https://www.scmp.com/tech/big-tech/article/3346789/alibaba-reshuffles-ai-units-new-token-hub-group-led-ceo-eddie-wu) [16]).

The financial commitment is staggering: Alibaba has pledged **RMB 380 billion (~\$53 billion)** for cloud and AI infrastructure over three years—nearly matching the company's total capital expenditure over the previous decade ([Alibaba Cloud, 2026](https://www.alibabacloud.com/blog/alibaba-to-invest-rmb380-billion-in-ai-and-cloud-infrastructure-over-next-three-years_602007) [19]). AI inference already drives 60–70% of new Alibaba Cloud revenue, with AI-related products achieving triple-digit growth for nine consecutive quarters [19].

The convergence between Huang's token factory vision and Alibaba's organizational restructuring is not coincidental. It reflects a broader industry consensus: **the business model of the AI era is shifting from software-as-a-service (SaaS) to tokens-as-a-service**. Instead of monthly subscription fees, enterprises will pay per volume of tokens processed—analogous to a utility bill for electricity. The Chinese-language press described ATH as Alibaba's play to capture **"pricing power and minting rights in the AI era" (AI时代铸币权)** [16]—positioning tokens as the central currency of the AI value chain.

> **Why this matters for business researchers:** The token economy redefines how value is created and captured in AI. The fundamental unit of production shifts from code (software) to tokens (AI inference). Competitive advantage depends on cost per token, token throughput, and the ability to orchestrate token-consuming agents at scale. Alibaba Token Hub, with its vertical integration from chips to models to consumer applications, exemplifies the emerging organizational form optimized for this new economy.

### 4.4. Approachability vs. Deep Work

OpenClaw crystallizes a fundamental tension in agent design: **approachability vs. deep work** ([Yage.ai deep dive](https://yage.ai/openclaw-en.html) [7]).

- **Chat-based interfaces** (WhatsApp, Telegram) are maximally approachable—everyone already knows how to send a message. But messaging apps impose severe constraints: no structured code editing, no version control integration, no persistent file system navigation.
- **IDE-based interfaces** (Claude Code, Cursor, VS Code) support deep, sustained technical work with full tooling. But they require technical sophistication and are inaccessible to non-developers.

OpenClaw chose approachability, betting that the messaging interface—despite its limitations for complex development tasks—would drive adoption by lowering the barrier to entry for the broadest possible user base.

### 4.5. Unified Context

A distinctive feature of OpenClaw is its **unified context pool** ([Yage.ai deep dive](https://yage.ai/openclaw-en.html) [7]; [OpenClaw GitHub](https://github.com/openclaw/openclaw) [8]): it mixes inputs from multiple messaging platforms (WhatsApp, Telegram, Discord, etc.) into a single data stream that the LLM agent can reason over. This means an instruction sent via WhatsApp can reference a document shared via Discord, and the agent maintains coherent context across all channels.

This raises a critical technical challenge: **how to deal with context window length limitations?** As conversations accumulate across multiple platforms, the token count grows rapidly. OpenClaw addresses this through its memory architecture (see Section 4.6) and aggressive context management.

### 4.6. Self-Evolving Memory Engine

OpenClaw implements a tiered, file-based memory architecture ([Yage.ai deep dive](https://yage.ai/openclaw-en.html) [7]; [OpenClaw GitHub](https://github.com/openclaw/openclaw) [8]):

| File | Purpose |
|------|---------|
| `SOUL.md` | Personality and behavioral guidelines—the agent's "character" |
| `USER.md` | User profile and preferences—learned over time |
| `MEMORY.md` | Long-term knowledge storage—facts, procedures, project state |

A background **heartbeat mechanism** automatically reviews recent interaction logs, distills valuable information into `MEMORY.md`, and cleans up outdated entries. This creates a *self-curating* knowledge base that evolves with each interaction.

The memory system is "blackboxed" in the sense that its internal state is opaque to the user—the agent decides what to remember and what to forget. This raises important questions about transparency and user control over agent behavior.

### 4.7. The OpenClaw Flywheel

The three core design elements—**persistent memory**, **unified context**, and **ecosystem of skills**—form a mutually reinforcing flywheel ([Yage.ai deep dive](https://yage.ai/openclaw-en.html) [7]; [OpenClaw GitHub](https://github.com/openclaw/openclaw) [8]):

```
          ┌──────────────────────┐
          │   Persistent Memory  │
          │  Retains learnings,  │
          │  habits, preferences │
          └────────┬─────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌────────────────┐
│ Unified       │    │ Ecosystem of   │
│ Context Pool  │    │ Skills         │
│ Mixes inputs  │    │ Emergent biz   │
│ from multiple │    │ capabilities & │
│ platforms     │    │ self-evolution  │
└───────┬───────┘    └───────┬────────┘
        │                    │
        └────────┬───────────┘
                 │
          (Reinforcing Loop)
```

- **Memory + Context:** The more platforms the agent monitors, the richer its memory becomes; richer memory improves contextual understanding across platforms.
- **Memory + Skills:** The agent can write its own skills—if no existing tool solves a problem, it writes code, saves it as a reusable skill, and adds it to its repertoire. This creates a self-evolution loop.
- **Skills + Context:** New skills unlock new capabilities, which generate new types of interactions, which feed back into context and memory.

These three subsystems achieve **continual learning**: the agent improves over time without explicit retraining, purely through accumulated experience and self-authored tooling.

### 4.8. The Lethal Trifecta

OpenClaw's power comes with proportional risk. The system exhibits what security researchers call the **lethal trifecta** of agentic AI (see the [Acronis security analysis](https://www.acronis.com/en/tru/posts/openclaw-agentic-ai-in-the-wild-architecture-adoption-and-emerging-security-risks/) [10]; [Yage.ai deep dive](https://yage.ai/openclaw-en.html) [7]; [OpenClaw GitHub](https://github.com/openclaw/openclaw) [8]):

1. **Access to private data:** The agent can read messages, files, browsing history, and other sensitive information across all connected platforms.
2. **Exposure to untrusted environments:** The agent processes inputs from the open internet, messaging platforms, and third-party skill repositories—all potential vectors for adversarial manipulation.
3. **Ability to take autonomous actions:** The agent can send messages, execute code, modify files, and interact with external services without human approval for every step.

When all three are present simultaneously, a single vulnerability can cascade into catastrophic compromise. This is not hypothetical: by mid-February 2026, multiple critical vulnerabilities had been disclosed:

- **CVE-2026-25253** (CVSS 8.8): Gateway compromise allowing arbitrary command execution.
- **CVE-2026-24763 and CVE-2026-25157:** Command injection vulnerabilities.
- **ClawHavoc:** A supply-chain poisoning campaign in which over 340 malicious skills were uploaded to [ClawHub](https://clawhub.ai/) [15] (the official skill marketplace), many posing as productivity tools but installing malware [10].

### 4.9. Reimplementation: Reconstructing the Magic Safely

OpenClaw's viral success demands massive compromises in security and control. However, its core innovations can be **reconstructed within a more controllable architecture** ([Yage.ai deep dive](https://yage.ai/openclaw-en.html) [7]). The reimplementation strategy involves four key decisions:

**1. Outsource the agentic loop to mature platforms.** Rather than building a custom agent runtime, leverage established coding agent platforms (Claude Code, Cursor, or similar) that have invested heavily in sandboxing, permission systems, and security hardening. These platforms provide the agentic loop—observation, reasoning, action, feedback—with enterprise-grade safeguards.

**2. Redesign the memory architecture.** Replace OpenClaw's blackboxed, opaque memory with a **disk-as-memory** approach backed by a **Git mono-repo**. This restores fine-grained control over knowledge assets and context:

| OpenClaw | Reimplementation |
|----------|-----------------|
| Opaque `MEMORY.md` managed by heartbeat | Version-controlled files in a Git repository |
| Blackboxed memory evolution | Transparent, auditable, diff-able history |
| Platform-mixed context | Workspace-isolated context per project |

**3. Mono-repo solves context pollution.** By organizing each project or workspace as a separate repository (or directory within a mono-repo), the agent's context is naturally scoped to the relevant domain. This prevents the "context pollution" problem where unrelated information from different platforms leaks into the agent's reasoning.

**4. Secure the skills supply chain.** Blindly installing third-party MCP servers risks catastrophic supply-chain attacks. The reimplementation approach favors writing skills in-house—a task that takes only minutes in the age of agentic AI coding—reducing execution risk to effectively zero. When third-party skills are necessary, they should be audited, sandboxed, and version-pinned.

> **Key lesson for researchers:** OpenClaw demonstrates that the *interface* and *memory architecture* of an agent system matter as much as the underlying LLM. The same model, wrapped in different scaffolding, can produce dramatically different user experiences, adoption curves, and risk profiles.

## 5. Context Engineering

### 5.1. Context Is the Key

**Context engineering** is the discipline of curating and maintaining the optimal set of information (tokens) delivered to a language model during inference (see [Lee, 2026, *The Agent Era*](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/agent_era.pdf) [11]; [Anthropic, *Effective Context Engineering for AI Agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]). It is the natural evolution of prompt engineering: while prompt engineering focuses narrowly on crafting effective instructions for specific tasks, context engineering addresses the *entire token management challenge* across multiple turns of inference in an agentic loop.

The distinction matters because agents operate in loops over extended periods. At each step, the agent must decide: *What information should be in the context window right now?* The universe of potentially relevant information—conversation history, tool outputs, retrieved documents, system instructions—grows monotonically, but the context window is finite. Context engineering is the art of managing this tension.

**Guiding principle:** Find the *smallest possible set of high-signal tokens* that maximizes the likelihood of the desired outcome ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]).

### 5.2. Why Context Engineering Matters: Context Rot

The theoretical basis for context engineering lies in the transformer's self-attention mechanism ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) [13]). Every token in the context window attends to every other token, creating $n^2$ pairwise relationships for $n$ tokens. As $n$ grows, the model's ability to capture these relationships gets stretched thin—a fundamental consequence of the quadratic attention architecture introduced by Vaswani et al. (2017) [13].

Empirical research on "needle-in-a-haystack" benchmarks has uncovered a phenomenon called **context rot** ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]): as the number of tokens in the context window increases, the model's ability to accurately recall and reason about information from that context *decreases*. This is not a cliff but a gradient—models remain capable at long contexts but show reduced precision for information retrieval and long-range reasoning compared to shorter contexts.

The implication is clear: **context is a finite resource with diminishing marginal returns**. Adding more information to the context window is not always better; it can actively hurt performance if the added tokens are low-signal or dilute attention from high-signal content.

### 5.3. Anatomy of Effective Context

[Anthropic's framework](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12] for effective context engineering identifies several key components:

**Agents as LLMs in a loop.** Anthropic defines agents as "LLMs autonomously using tools in a loop." This minimal definition emphasizes that the agentic loop—not any particular architecture or framework—is the essential structure. Smarter models allow agents to more independently navigate nuanced problem spaces and recover from errors, but the loop structure is universal.

**The three pillars of context design:**

1. **System prompts** define the agent's altitude—how it should approach tasks, what constraints apply, what persona it should adopt.
2. **Tools** define the strict contract between agents and their information/action space—what the agent *can* do and how it learns about the world.
3. **Examples** (few-shot in-context learning) communicate behavioral patterns more efficiently than verbose rules—they are the "pictures worth a thousand words" for LLMs.

### 5.4. System Prompts: Finding the Right Altitude

System prompts define the *altitude* at which an agent operates ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]). Two failure modes must be avoided:

- **Overly specific (too low):** Hardcoding complex, brittle if-else logic that attempts to enumerate every edge case. This creates fragile systems that break at the first unanticipated input.
- **Overly vague (too high):** Providing vague guidance that assumes the model shares the developer's implicit understanding. This leads to inconsistent, unpredictable behavior.

The optimal altitude is **specific enough to guide behavior effectively, yet flexible enough to provide strong heuristics** that generalize across situations.

**Best practices for system prompt design:**

- Organize prompts into distinct sections using XML tags or Markdown headers:
  - `<background_information>` — domain context and objectives
  - `<instructions>` — behavioral guidelines and constraints
  - `### Tool guidance` — when and how to use specific tools
  - `## Output description` — format and structure of expected outputs
- Start with a minimal prompt on the strongest available model.
- Iteratively add clarifications based on observed failure modes—never preemptively over-specify.
- Use canonical examples rather than exhaustive rule lists. A curated set of diverse examples that portray expected behavior is more effective than paragraphs of instructions.

### 5.5. Token-Efficient Tool Design

Tools define the contract between agents and their action space. The design of tools has a direct impact on agent performance and token efficiency ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]):

- **Self-contained and robust to error.** Each tool should handle edge cases gracefully and return informative error messages.
- **Clear intended use with minimal overlap.** If a human engineer cannot definitively say which tool applies in a given scenario, an AI agent cannot do better. Bloated toolsets with overlapping functionality are a common failure mode.
- **Descriptive, unambiguous input parameters.** Use `user_id` rather than `user`; `start_date` rather than `date`.
- **Token-efficient outputs.** Implement pagination, filtering, and truncation with sensible defaults for any tool that could return large responses. A tool that dumps 10,000 tokens of raw data when the agent needs a 50-token summary is wasting the context budget.

The key insight is that tools should be designed as if writing documentation for a new hire: **make all implicit knowledge explicit** (specialized query formats, niche terminology, resource relationships).

### 5.6. Progressive Information Retrieval vs. RAG

A critical design choice for agentic systems is *how* the agent obtains information ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]).

**Traditional RAG (Retrieval-Augmented Generation):** Embedding-based retrieval surfaces all potentially relevant context *upfront*, before the model begins generating. This is fast but rigid—the retrieval query is formulated before the model has had a chance to reason about what it actually needs.

**Just-in-time agentic retrieval:** The agent maintains lightweight identifiers (file paths, stored queries, URLs) and dynamically loads data using tools *at runtime*. This mirrors human cognition—we do not memorize entire libraries but rather develop indexing systems (file folders, bookmarks, mental models of where to look) that enable efficient on-demand retrieval.

| Dimension | Traditional RAG | Just-in-Time Retrieval |
|-----------|:---------------:|:---------------------:|
| Speed | Faster (pre-computed) | Slower (runtime exploration) |
| Flexibility | Less adaptive | Highly adaptive |
| Context efficiency | May over-retrieve | Retrieves only what is needed |
| Staleness risk | Index may be outdated | Always reads current data |
| Complexity | Simpler pipeline | Requires thoughtful tool design |

**The hybrid approach (exemplified by Claude Code):**
- Pre-load high-signal static information (e.g., `CLAUDE.md` files) into context upfront.
- Use targeted tools (`glob`, `grep`, `read`) for just-in-time file navigation and content retrieval.
- Bypass issues of stale indexing and complex syntax trees.

This hybrid strategy strikes a practical balance: immediate context for known-relevant information, combined with autonomous exploration for dynamic or large-scale content.

### 5.7. Architecting for Long Horizons

Long-horizon tasks—those requiring sustained agent effort over many turns of the agentic loop—present a fundamental challenge: the token count of accumulated context eventually approaches or exceeds the model's context window. Three complementary mechanisms address this ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]):

#### 5.7.1. Compaction: Distilling the State

**Compaction** involves summarizing a conversation that has reached context limits and restarting with the compressed summary. It is the most direct lever for extending agent coherence over long interactions.

**Implementation pattern (as in Claude Code):**
1. Pass the full message history to the model for summarization.
2. Preserve architectural decisions, unresolved bugs, and critical implementation details.
3. Discard redundant tool outputs, stale intermediate results, and superseded reasoning.
4. Continue with compressed context plus the most recently accessed files.

**The art of compaction** lies in selecting what to preserve versus what to discard. Overly aggressive compaction risks losing subtle but critical context whose importance only becomes apparent later. The recommended approach:
- **Maximize recall first:** Capture everything that might be relevant.
- **Iterate to improve precision:** Eliminate superfluous content through testing and observation.

A lightweight form of compaction that is nearly always safe: **clearing old tool results** from message history. Agents rarely need the raw output of tools called deep in the conversation—the *conclusions* drawn from those outputs, already embedded in the agent's subsequent reasoning, are sufficient.

#### 5.7.2. Agentic Memory: Structured Note-Taking

Rather than relying solely on the context window, agents can **write externally-persisted notes** that are retrieved at later times ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]). This provides persistent memory with minimal context overhead.

**Patterns include:**
- **To-do lists:** Tracking progress across complex, multi-step tasks.
- **Notes files:** Maintaining `NOTES.md` or similar structured documents with critical dependencies, decisions, and status.
- **Memory files:** Building knowledge bases over time that persist across sessions.

The structured note-taking approach is particularly effective for iterative development with clear milestones. The agent writes notes at natural checkpoints, and future turns (or even future sessions) can read these notes to reconstruct context without replaying the full conversation history.

> **Example:** Claude playing Pokémon maintains precise tallies across thousands of game steps—tracking training progress, discovered areas, and strategic combat notes. After context resets, the agent reads its own notes and continues multi-hour gameplay sequences seamlessly ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]).

#### 5.7.3. Sub-Agent Architectures

Rather than one agent maintaining state across an entire project, **specialized sub-agents** handle focused tasks with clean context windows ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]):

- A **lead (orchestrator) agent** coordinates at a high level: planning, delegating, and synthesizing.
- **Sub-agents** perform deep technical work—research, analysis, code generation—within their own isolated context windows.
- Each sub-agent may consume tens of thousands of tokens in exploration but returns only a condensed 1,000–2,000 token summary to the orchestrator.

This architecture provides:
- **Clear separation of concerns:** Detailed search context remains isolated within sub-agents.
- **Context efficiency:** The lead agent's context window contains only high-level plans and sub-agent summaries, not the raw details of every exploration.
- **Parallelism:** Multiple sub-agents can work simultaneously on independent sub-tasks.

Research on multi-agent research systems has shown substantial improvements over single-agent approaches on complex analysis tasks ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]).

**Choosing the right mechanism:**

| Mechanism | Best For |
|-----------|---------|
| Compaction | Tasks requiring extensive back-and-forth interaction |
| Structured note-taking | Iterative development with clear milestones |
| Sub-agent architectures | Complex research/analysis where parallel exploration pays dividends |

### 5.8. Context Engineering Capability Map

The full context engineering capability map can be organized along two dimensions ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]):

1. **Temporal dimension:** From single-turn static context design (system prompts, tools, examples) through multi-turn dynamic retrieval to long-horizon context management (compaction, memory, sub-agents).

2. **Autonomy dimension:** From engineer-curated static context (manually written prompts and examples) through hybrid approaches (pre-loaded context plus autonomous retrieval) to fully autonomous agent-driven exploration.

```
                 Static Context          Dynamic Retrieval       Long-Horizon Management
                 ─────────────           ─────────────────       ───────────────────────
Engineer-      │ System prompts,      │ Traditional RAG,       │ Manual checkpointing,
Curated        │ tool definitions,    │ pre-computed indices   │ session summaries
               │ few-shot examples    │                        │
               │                      │                        │
Hybrid         │ CLAUDE.md files,     │ Hybrid retrieval       │ Compaction with
               │ structured configs   │ (static + tools)       │ preserved landmarks
               │                      │                        │
Fully          │ Agent-generated      │ Just-in-time           │ Sub-agents, memory,
Autonomous     │ prompts and tools    │ agentic exploration    │ self-evolving notes
```

As models become more capable, the optimal point on this map shifts toward greater autonomy—but thoughtful context curation remains central regardless of model intelligence.

### 5.9. Formal Representation of Context Engineering

Context engineering can be expressed in the following abstract format ([Lee, 2026](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/agent_era.pdf) [11]; [Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [12]). Let $P_t$ denote the set of information (tokens) delivered to the context window of the LLM at time step $t$ of the agentic loop. The agent's behavior at each step is determined by:

$$\text{action}_t = \text{LLM}(P_t)$$

where $P_t$ includes the system prompt, tool definitions, conversation history (possibly compacted), retrieved information, memory contents, and any other tokens present in the context window.

The **context engineering problem** is to design a policy for constructing $P_t$ that maximizes the probability of achieving the desired outcome:

$$\max_{P_t} \; \Pr(\text{desired outcome} \mid P_t) \quad \text{subject to} \quad |P_t| \leq C$$

where $C$ is the context window size (in tokens) and $|P_t|$ denotes the token count of $P_t$.

The constraint $|P_t| \leq C$ is hard—exceeding the context window causes truncation or failure. But the optimization is also subject to a *soft* constraint: **context rot** implies that the effective information capacity of the context window is less than $C$. Adding low-signal tokens to $P_t$ can *reduce* $\Pr(\text{desired outcome} \mid P_t)$ even if $|P_t| < C$, because they dilute attention from high-signal content.

The context window grows easily—every observation, tool output, and reasoning step adds tokens—but the information budget is fixed. The central challenge of context engineering is managing this asymmetry: an ever-expanding universe of potentially relevant information filtered through a finite, degradation-prone window.

> **The key question:** How should we properly design $P_t$?

This question does not have a universal answer. The optimal $P_t$ depends on the task, the model, the available tools, and the interaction history. Context engineering is as much an empirical art as a theoretical discipline—and it is the defining skill for building effective AI agents.

## 6. Conclusion

Agentic AI represents the next frontier in applied artificial intelligence—the transition from models that generate text to systems that autonomously plan, act, and learn in the world. The key themes of this lecture are:

1. **From LLMs to agents.** Equipping an LLM with an agentic loop—observation, reasoning, action, feedback—transforms it from a passive text generator into an active problem-solver. The critical enablers are tool use, memory, and task decomposition.

2. **Skills and MCP as infrastructure.** Skills provide modular SOPs that make agents effective at specific tasks. Three archetypes—document creation, workflow automation, and MCP enhancement—cover the primary use cases. The progressive disclosure architecture ensures token efficiency by loading skill content in three tiers (metadata, instructions, resources). Clear frontmatter descriptions drive reliable triggering, and nonambiguous instructions bridge the gap between deterministic code and probabilistic language. The [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) standardizes how agents connect to external tools and data, solving the $M \times N$ integration problem with an open, universal standard.

3. **OpenClaw and the token economy.** [OpenClaw](https://openclaw.ai/)'s viral success demonstrates the power of approachability (messaging-first interface), unified context (multi-platform data integration), and the flywheel of memory + skills + context. But it also exposes the lethal trifecta of agentic risk: private data access, untrusted input exposure, and autonomous action capability. More broadly, the emergence of agentic AI has catalyzed the **token economy**—Jensen Huang's vision of data centers as "token factories" producing inference as a commodity—with [Alibaba Token Hub](https://www.alizila.com/alibaba-establishes-alibaba-token-hub-business-group/) as a landmark organizational response: a top-level business group built around the mission to "create tokens, deliver tokens, and apply tokens."

4. **Context engineering as the defining discipline.** Building effective agents is less about model intelligence and more about the quality of information delivered to the model at each step. [Context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)—encompassing system prompt design, token-efficient tool definitions, progressive retrieval strategies, compaction for long horizons, structured note-taking for persistent memory, and sub-agent architectures for parallel exploration—is the skill that separates functional agents from unreliable ones.

5. **The formal challenge.** Context engineering can be framed as an optimization problem: maximize the probability of the desired outcome subject to the hard constraint of context window size and the soft constraint of context rot. The ever-expanding universe of relevant information, filtered through a finite and degradation-prone window, defines the central tension that practitioners must navigate.

> **The broader significance for business researchers:** Agentic AI is already transforming how research is conducted—from automated literature reviews and data analysis to experimental design and paper writing. Understanding the architecture of agents (loops, tools, memory), the standards that connect them to the world ([MCP](https://modelcontextprotocol.io/)), the risks they introduce (the lethal trifecta), and the principles that make them effective ([context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)) is essential for any researcher who wishes to leverage—or study—these systems.

## References

[1] Song, D., Chen, X., & Yang, K. (2025). *Language Agent Tutorial.* UC Berkeley. [https://language-agent-tutorial.github.io/](https://language-agent-tutorial.github.io/); Slides: [https://language-agent-tutorial.github.io/slides/I-Introduction.pdf](https://language-agent-tutorial.github.io/slides/I-Introduction.pdf)

[2] Song, D., Chen, X., & Yang, K. (2025). *Advanced LLM Agents MOOC.* UC Berkeley. [https://llmagents-learning.org/sp25](https://llmagents-learning.org/sp25); Slides: [https://llmagents-learning.org/slides/intro.pdf](https://llmagents-learning.org/slides/intro.pdf)

[3] Lee, H. (2026). *Machine Learning Course (Spring 2026).* National Taiwan University. [https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/intro.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/intro.pdf)

[4] Anthropic. (2025). *The Complete Guide to Building Skills for Claude.* [https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf); Skills repository: [https://github.com/anthropics/skills](https://github.com/anthropics/skills)

[5] DeepLearning.AI. (2025). *Agent Skills with Anthropic.* Short Course. [https://www.deeplearning.ai/short-courses/agent-skills-with-anthropic/](https://www.deeplearning.ai/short-courses/agent-skills-with-anthropic/)

[6] Anthropic. (2024). *Introducing the Model Context Protocol.* [https://www.anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol); Protocol specification: [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/); Architecture: [https://modelcontextprotocol.io/docs/learn/architecture](https://modelcontextprotocol.io/docs/learn/architecture)

[7] Yage.ai. (2026). *OpenClaw Deep Dive.* [https://yage.ai/openclaw-en.html](https://yage.ai/openclaw-en.html)

[8] OpenClaw. (2026). *Official Website and Documentation.* [https://openclaw.ai/](https://openclaw.ai/); GitHub: [https://github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)

[9] NVIDIA. (2026). *GTC Keynote.* [https://www.nvidia.com/gtc/keynote/](https://www.nvidia.com/gtc/keynote/)

[10] Acronis. (2026). *OpenClaw: Agentic AI in the Wild—Architecture, Adoption, and Emerging Security Risks.* [https://www.acronis.com/en/tru/posts/openclaw-agentic-ai-in-the-wild-architecture-adoption-and-emerging-security-risks/](https://www.acronis.com/en/tru/posts/openclaw-agentic-ai-in-the-wild-architecture-adoption-and-emerging-security-risks/)

[11] Lee, H. (2026). *The Agent Era.* National Taiwan University. [https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/agent_era.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/agent_era.pdf)

[12] Rajasekaran, P., Dixon, E., Ryan, C., & Hadfield, J. (2025). *Effective Context Engineering for AI Agents.* Anthropic Engineering Blog. [https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

[13] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is all you need.* Advances in Neural Information Processing Systems (NeurIPS), 30. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

[14] Steinberger, P. (2026). *OpenClaw and OpenAI.* Personal Blog. [https://steipete.me/posts/2026/openclaw](https://steipete.me/posts/2026/openclaw)

[15] ClawHub. (2026). *Skills Marketplace.* [https://clawhub.ai/](https://clawhub.ai/); AgentSkills.io: [https://agentskills.io/home](https://agentskills.io/home)

[16] South China Morning Post. (2026). *Alibaba reshuffles AI units into new Token Hub group led by CEO Eddie Wu.* [https://www.scmp.com/tech/big-tech/article/3346789/alibaba-reshuffles-ai-units-new-token-hub-group-led-ceo-eddie-wu](https://www.scmp.com/tech/big-tech/article/3346789/alibaba-reshuffles-ai-units-new-token-hub-group-led-ceo-eddie-wu)

[17] Alizila. (2026). *Alibaba establishes Alibaba Token Hub business group.* [https://www.alizila.com/alibaba-establishes-alibaba-token-hub-business-group/](https://www.alizila.com/alibaba-establishes-alibaba-token-hub-business-group/)

[18] TechNode. (2026). *Alibaba Group forms Alibaba Token Hub unit; CEO Eddie Wu to lead AI push.* [https://technode.com/2026/03/17/alibaba-group-forms-alibaba-token-hub-unit-ceo-eddie-wu-to-lead-ai-push/](https://technode.com/2026/03/17/alibaba-group-forms-alibaba-token-hub-unit-ceo-eddie-wu-to-lead-ai-push/)

[19] Alibaba Cloud. (2026). *Alibaba to invest RMB 380 billion in AI and cloud infrastructure over next three years.* [https://www.alibabacloud.com/blog/alibaba-to-invest-rmb380-billion-in-ai-and-cloud-infrastructure-over-next-three-years_602007](https://www.alibabacloud.com/blog/alibaba-to-invest-rmb380-billion-in-ai-and-cloud-infrastructure-over-next-three-years_602007)
