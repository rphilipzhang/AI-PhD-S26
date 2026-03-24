# What's New in AI: Lecture Notes

**DOTE 6635: Artificial Intelligence for Business Research (Spring 2026)**

**Instructor: Renyu (Philip) Zhang**

*These notes accompany the "What's New in AI" module, covering rapid developments in artificial intelligence from January to March 2026. The material is curated for PhD students across business disciplines --- from operations and marketing to finance, strategy, and information systems. Each topic is presented with research implications for business scholars.*

---

## Table of Contents

1. [The AI Landscape in Early 2026: A Bird's-Eye View](#1-the-ai-landscape-in-early-2026-a-birds-eye-view)
2. [AI Coding and the Zero Marginal Cost of Code](#2-ai-coding-and-the-zero-marginal-cost-of-code)
3. [AI-Driven Scientific Replication and Reproducibility](#3-ai-driven-scientific-replication-and-reproducibility)
4. [AI for Mathematics: From Putnam to Unsolved Problems](#4-ai-for-mathematics-from-putnam-to-unsolved-problems)
5. [AI and the Labor Market: Displacement, Augmentation, and the Intelligence Crisis](#5-ai-and-the-labor-market-displacement-augmentation-and-the-intelligence-crisis)
6. [AI Assistance vs. Human Learning](#6-ai-assistance-vs-human-learning)
7. [Agentic AI: Social Networks, Commerce, and Workflows](#7-agentic-ai-social-networks-commerce-and-workflows)
8. [The Business of AI: Economics, IPOs, and Open vs. Closed Models](#8-the-business-of-ai-economics-ipos-and-open-vs-closed-models)
9. [AI in High-Stakes Domains: Medicine, Law, and Policy](#9-ai-in-high-stakes-domains-medicine-law-and-policy)
10. [Organizational Transformation in the AI Age](#10-organizational-transformation-in-the-ai-age)
11. [Ethics, Privacy, and Governance](#11-ethics-privacy-and-governance)
12. [Towards AGI: Self-Improvement and the Road Ahead](#12-towards-agi-self-improvement-and-the-road-ahead)
13. [Discussion Questions for PhD Researchers](#13-discussion-questions-for-phd-researchers)
14. [References](#14-references)

---

## 1. The AI Landscape in Early 2026: A Bird's-Eye View

### 1.1 The Technology Tree Is Growing Fast

This course (DOTE 6635) has been offered every spring since 2024, and its evolution tells the story of AI itself. In its first year, the focus was on natural language processing and computer vision. In Year 2, it shifted to large language models and causal machine learning. Now, in Year 3, the centerpiece is **reinforcement learning and agentic AI**. The fact that the syllabus cannot repeat prior content for more than a fraction of each offering speaks to the velocity of the field: the technology tree is growing faster than any curriculum can track.

This pace is reminiscent of a motivation drawn from Demis Hassabis, co-founder of Google DeepMind and 2024 Nobel Laureate in Chemistry. As a 12-year-old chess prodigy, Hassabis resigned a drawn position against the ex-Danish champion out of sheer exhaustion --- then immediately realized his error when his opponent pointed out the stalemate. The experience prompted a question that would shape his career: *Are we wasting our minds? Is this the best use of all this brain power?* The same question animates this course: in an era when AI can handle an increasing share of cognitive labor, where should brilliant human minds focus?

### 1.2 2025 in Review: Nature's Ten and Karpathy's Reflections

**Liang Wenfeng and Nature's Ten.** At the close of 2025, *Nature* named **Liang Wenfeng**, the founder of DeepSeek, to its annual list of ten people who shaped science. *Nature* called him a "tech disruptor" whose open-source models demonstrated that the U.S. was not as far ahead in AI as many experts had assumed ([Nature, Dec 2025](https://www.nature.com/articles/d41586-025-03845-4); [SCMP, Dec 2025](https://www.scmp.com/tech/tech-trends/article/3335748/deepseeks-liang-wenfeng-makes-cut-top-10-people-who-shaped-science-2025)). For context, the 2023 list featured ChatGPT and Ilya Sutskever. The shift from an American product to a Chinese entrepreneur signals a meaningful rebalancing of the global AI landscape.

A related milestone: **DeepSeek Math-V2** became the first open-source model to achieve IMO gold medal-level performance, solving 5 of 6 problems at the 2025 International Mathematical Olympiad. The key engineering insight was **self-verification** --- the model checks its own proofs for rigor and completeness, a dual-model architecture that mimics how human mathematicians work ([SCMP, 2025](https://www.scmp.com/tech/tech-trends/article/3334553/deepseek-releases-first-open-ai-model-gold-level-scores-maths-olympiad); [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-Math-V2)).

**Karpathy's Year-in-Review.** Andrej Karpathy's widely circulated blog post ([Karpathy, 2025](https://karpathy.bearblog.dev/year-in-review-2025/)) provides a useful snapshot of where the field stood at the turn of 2026. Several themes stand out:

- **RLVR (Reinforcement Learning with Verifiable Rewards)** emerged as the most consequential technical development of 2025, fundamentally altering the LLM training stack. Much of the capability progress came from longer RL runs rather than larger pretraining datasets.
- **The shape of intelligence differs.** Karpathy offered an evocative analogy: human intelligence is like an irregularly shaped star --- strong in some directions, weak in others. AI intelligence has a *different* shape. The overlap is partial, which is precisely why human-AI coordination (and **context engineering**) matters so much.
- **Cursor and "vibe coding."** Tools like Cursor dramatically shortened the distance between human intent and compute. The term "vibe coding" --- writing software by describing what you want in natural language --- entered the mainstream vocabulary.
- **Summary:** Karpathy concluded that 2025 was "an exciting and mildly surprising year" where LLMs proved simultaneously smarter and dumber than expected.

### 1.3 The Four Maintracks of AI Progress

The Spring 2026 AI landscape can be organized around four maintracks --- all of which share a common thread: **facilitating humans to leverage compute**. Human brains are good at many things, but raw computation is not one of them. GPUs are. The state-of-the-art AI stack is fundamentally about bridging that gap.

These four tracks are listed in *increasing order of importance but decreasing order of maturity*:

| Maintrack | Core Idea | Business Research Implication |
|---|---|---|
| **AI Coding** | Connecting human intent with compute through natural language | Productivity, innovation, software economics |
| **Deep Research** | Automating information acquisition and processing | Knowledge work transformation, research methodology |
| **World Models** | Data-driven simulation of the physical world | Operations, supply chain, digital twins |
| **AI Scientist** | Automated hypothesis generation and validation | Research design, scientific discovery, R&D strategy |

To make these concrete:

- **AI Coding** is the most mature. Previously, building software from scratch required a team of expert engineers. Now, with vibe coding, a single person can produce a working prototype in hours --- 10x to 100x faster.
- **Deep Research** is the automation of information acquisition and processing. If you use ChatGPT's deep research function to do a literature review today, it will, with high probability, produce a more comprehensive survey than most individual researchers could. The limitation: AI still lacks the *judgment* to evaluate what matters.
- **World Models** use data-driven simulation to replicate physical environments, providing feedback loops for AI systems in robotics, autonomous vehicles, and operations.
- **AI Scientist** represents the frontier: automated hypothesis generation, validation, and iteration. When this maintrack matures, it will fundamentally reshape --- and potentially displace --- the work of academic researchers.

These four tracks reflect a paradigm shift: AI is no longer just a tool for prediction (the "ML for X" era), but an **active agent** that can plan, execute, and iterate on complex tasks.

### 1.4 The AI Industry by the Numbers

A few data points capture the scale of what is happening:

- **The capital cycle:** Nvidia invested $100 billion over the next few years. OpenAI uses the money to purchase Oracle's cloud services. Oracle then purchases Nvidia's chips. The money circulates, but the infrastructure grows relentlessly.
- **Talent wars:** Meta's Mark Zuckerberg aggressively recruited top AI talent, reportedly offering Yu Jiahui (from OpenAI) an annual package of ~$100 million. By contrast, an economist position at OpenAI --- still generous by academic standards --- pays orders of magnitude less. The market is pricing foundational AI skills at a premium that dwarfs adjacent fields.
- **AI-generated content:** By some estimates, roughly half of new internet content (on text-heavy platforms) is now AI-generated. For video platforms like YouTube and Bilibili, the share is lower but rising fast.
- **Meta acquired Manus**, a Chinese-founded agentic AI startup based in Singapore, for over $2 billion in late December 2025 ([CNBC, Dec 2025](https://www.cnbc.com/2025/12/30/meta-acquires-singapore-ai-agent-firm-manus-china-butterfly-effect-monicai.html)). Notably, none of the Manus founders were building foundational AI --- they were "wrappers" who figured out the right market application. The lesson: in the age of abundant AI capabilities, **market insight and product sense** can be as valuable as technical breakthroughs.

### 1.5 Key Model Launches

The first quarter of 2026 witnessed several landmark releases:

- **Codex-5.3 and Claude Opus-4.6** launched within 10 minutes of each other in February 2026, marking a new era of **agentic engineering** --- models that can not only generate code but autonomously manage multi-step software development workflows ([Karpathy, Feb 2026](https://x.com/karpathy/status/2019137879310836075)). The instructor noted that the upgrade from their predecessors (Codex-2.5.2 and Claude Opus-4.5) was "still very noticeable, even from a user's experience" --- and critically, these models **use their own code to upgrade themselves**, creating a flywheel of self-improvement that never stops. Karpathy coined the term "agentic engineering" for this paradigm, just as he had coined "vibe coding" the previous year. The practical implication: the tools are evolving faster than any course can track, and nobody yet knows how to leverage them to their full potential.
- **DeepSeek V4** generated significant anticipation around Chinese New Year --- exactly one year after DeepSeek R1 shocked the world. Multiple sources, including subscription-only outlets like *The Information*, cited insiders claiming V4 could match or exceed Claude's coding performance. Reported features included dramatically improved context understanding for complex coding prompts, sustained scaling-law gains (no performance degradation with more data and compute), and stronger code reasoning capabilities ([Yahoo Tech, Jan 2026](https://tech.yahoo.com/ai/articles/insiders-deepseek-v4-beat-claude-205234497.html)). *Note: As of March 2026, DeepSeek V4 has not officially launched despite multiple predicted release windows. The claims remain unverified by independent benchmarks.*
- **Google Aletheia**, a system designed to autonomously verify mathematical proofs, tackled problems from the First Proof initiative ([Luong et al., 2026](https://arxiv.org/pdf/2602.21201)).

> **For business researchers:** These model launches are not mere engineering milestones. Each represents a shift in the cost structure of cognitive labor, with direct implications for organizational design, market structure, and competitive strategy.

---

## 2. AI Coding and the Zero Marginal Cost of Code

### 2.1 The AI-Native Cost Structure

One of the most consequential ideas in early 2026 is the notion of **zero marginal cost of code**. When coding agents can autonomously generate, test, and deploy software, the economics of software production fundamentally change. This is the "AI-native mindset" --- treating code not as a scarce, expensive artifact but as an abundant, near-zero-cost commodity ([Yage.ai, 2026](https://yage.ai/ai-native-cost-structure.html); [The Modern Software Dev, 2026](https://themodernsoftware.dev/)).

A vivid illustration: in January 2026, **Michael Truell**, CEO of Cursor, orchestrated hundreds of GPT-5.2 agents to autonomously build a web browser ("FastRender") from scratch --- 3 million+ lines of Rust code including an HTML parser, CSS cascade engine, layout system, and custom JavaScript VM. It ran uninterrupted for **one week** with no human instructions. The result was far from production-ready (critics called it "shoddy code at scale"), but the signal was unmistakable: a project that would have taken a team of engineers several months was executed in a single autonomous run ([Truell, Jan 2026](https://x.com/mntruell/status/2011562190286045552); [Fortune, Jan 2026](https://fortune.com/2026/01/23/cursor-built-web-browser-with-swarm-ai-agents-powered-openai/)).

The practical implication, as emphasized repeatedly in the lecture: code is now **disposable**. If a piece of code can only be used once, that is fine --- GPUs and storage are cheap relative to the value of a researcher's time. As an OpenAI infrastructure engineer put it in a widely circulated podcast: "It is much easier to teach an engineer how to do research than to teach a scientist how to do engineering. The most critical thing is iteration speed." The corollary for PhD students: let your AI agents work around the clock. If they are idle, you are wasting your most precious resource --- time.

**Implications for business research:**

- **Platform economics:** If the cost of building software approaches zero, barriers to entry collapse. What sustains competitive advantage when anyone can spin up a product?
- **Innovation strategy:** The bottleneck shifts from "can we build it?" to "should we build it?" --- from engineering capacity to taste, judgment, and market understanding.
- **Operations management:** Customized software solutions for niche operational problems become feasible, enabling mass customization of business processes.

### 2.2 Quality of AI-Generated Code

However, zero marginal cost does not imply zero marginal risk. A rigorous audit by Graham Straus and Andrew Hall examined how accurately Claude Code replicated and extended Hall's published PNAS paper on vote-by-mail ([Straus & Hall, 2026](https://www.andrewbenjaminhall.com/Straus_Hall_Claude_Audit.pdf)). The audit found both impressive successes and instructive failure modes.

**What went right:**
- Claude replicated the original paper's estimates exactly and coded 29 of 30 California counties correctly on treatment timing.
- The collected election data correlated above **0.999** with manually collected ground-truth data.
- Overall, the AI did a "remarkably good job" --- the estimates were similar in magnitude to human-produced results.

**What went wrong:**
- The main mistake was a **failure to collect all needed data** --- specifically, senatorial and gubernatorial election data for two states. This is a judgment error (knowing *what* data to gather), not a coding error.
- One county's treatment year was miscoded, and non-presidential elections were not used to compute turnout --- subtle errors of the kind a human RA might also make on a first pass.
- AI tends to produce **unsolicited extensions** --- additional analyses and robustness checks that were not requested and lack clear academic value. As the instructor noted, AI is eager to "do more" but lacks the judgment to know what additions are scientifically meaningful.

**The exponential trajectory.** The instructor emphasized that AI coding capability is improving at an exponential rate: what was 1x in 2024 became roughly 10x in 2025 and is on track for 100x in 2026. The practical advice: don't compete with AI on execution --- focus on judgment, taste, and knowing *what* to ask. Document your AI-assisted workflow thoroughly, because transparency about the process is more valuable than hiding it.

> **Takeaway:** The productivity gains from AI coding are real, but they demand a new form of literacy: the ability to **audit and validate** machine-generated code. For empirical researchers, this is not optional --- it is a matter of scientific integrity. AI excels at execution but still struggles with the judgment calls that define good research: what data to collect, which specifications matter, and when to stop extending.

---

## 3. AI-Driven Scientific Replication and Reproducibility

### 3.1 Complete Replication of a PNAS Paper

In one of the most striking demonstrations of early 2026, Professor Andrew Hall of Stanford Graduate School of Business used Claude Code to achieve a **complete replication** of his previously published PNAS paper --- Thompson, Wu, Yoder, and Hall (2020), "Universal Vote-by-Mail Has No Impact on Partisan Turnout or Vote Share" ([PNAS](https://www.pnas.org/doi/10.1073/pnas.2007249117)). The original study used a staggered difference-in-differences design to examine how universal vote-by-mail in Washington, Utah, and California (the only three U.S. states that adopted it) affected partisan electoral outcomes. The key findings: vote-by-mail does not affect either party's share of turnout or vote share, but modestly increases overall average turnout rates. The entire AI-driven replication process, including all prompts and conversations, was open-sourced ([Hall, 2026](https://github.com/andybhall/vbm-replication-extension)).

**How it was done.** Hall used **Claude Opus 4.5** via the command-line interface (functionally equivalent to VS Code or Cursor integrations). Unlike the single-figure demo in Section 3.2, this was a *full paper* replication: literature review, data collection, original analysis, data extension through 2024, robustness checks, and paper writing. Hall laid out a series of **checkpoints** --- at each milestone, the AI summarized its progress and flagged concerns for human review before proceeding. The remarkable finding: the process was essentially **one-shot**. Most human prompts were simply "Approved, please go ahead." The AI read the original paper, understood the methodology, collected and processed data, ran the analysis, and produced a complete replication with extension --- with minimal human correction along the way. Hall subsequently published a detailed assessment of the replication's accuracy ([Hall, 2026](http://www.andrewbenjaminhall.com/)).

**Cost.** The basic Claude Code subscription is approximately $28/month. The entire replication of a PNAS paper cost less than a single month's subscription.

This raises profound questions:
- If an AI agent can replicate a paper from its instructions and data, what does that say about the **complexity** (or lack thereof) of much empirical research?
- Can AI-driven replication become a **standard part of the peer-review process**?
- What happens to the value of "execution skill" when execution is automated?

> **Practical advice from the instructor:** Open-source your AI-driven replication, but go beyond "I asked AI to do it and it finished." Share the insights --- what worked, what failed, what surprised you. The learning is in the process, not just the output.

### 3.2 Live Demo: Replicating a Published Figure in 10 Minutes

In the opening lecture, a live demonstration illustrated the paradigm shift. The task: replicate a key figure from a published methodology paper on Double Machine Learning (DML), which shows how cross-validation error decreases across training epochs and how the proposed method outperforms a baseline. Normally, this is the kind of exercise given to a new PhD student or research assistant --- read the paper, understand the methodology, write synthetic data generation code, train deep neural networks, and produce the figure. It typically takes days.

The procedure:
1. Write a short instruction file (the same instructions one would give a human RA).
2. Provide the paper PDF and the original figure for reference.
3. Run a single prompt in a coding agent (Codex / Claude Code).

Within **10 minutes**, the agent: read and summarized the paper's methodology, generated synthetic data, implemented the DML training pipeline, computed the benchmarks, and produced a figure that closely replicated the original --- not perfectly, but to a standard that would pass muster for an RA's first attempt. The entire process, including prompts, is available on GitHub for students to reproduce.

> **Implication:** The exercise that used to be a multi-day proving ground for new researchers can now be completed in minutes. This does not make the researcher obsolete --- it shifts the bottleneck from *execution* to *judgment*: knowing what to replicate, why it matters, and whether the output is correct.

### 3.3 Scaling Reproducibility

A broader initiative to **scale reproducibility** was introduced by **Yiqing Xu** (Stanford Political Science) and **Leo Y. Yang** (Hong Kong Baptist University) in their paper "Scaling Reproducibility: An AI-Assisted Workflow for Large-Scale Reanalysis" ([Xu & Yang, 2026](https://arxiv.org/abs/2602.16733); [Video Demo](https://www.youtube.com/watch?v=lhSIOPSxKc0)). The system uses a **Claude Code agentic architecture** to automatically reproduce results from published empirical papers.

**Scope and results.** The system was tested on **92 instrumental variable (IV) studies** drawn from three top political science journals --- the *American Political Science Review* (APSR), the *American Journal of Political Science* (AJPS), and *The Journal of Politics* (JOP) --- spanning 2010--2025. The headline result: **87% overall end-to-end success rate** (55 of 67 benchmark papers plus all 25 newly published papers). When authors provided both code and data, reproducibility was **100%** at both the paper and specification levels, across 215 total specifications.

**The three-layer architecture.** As the instructor explained, the system employs a three-level pipeline:

1. **Orchestration layer.** An LLM orchestrator coordinates the overall workflow, receiving human inputs (the research paper, basic code, identification strategy, data documentation, README files) and directing specialized sub-agents.
2. **Skills layer.** Structured instructions --- essentially standard operating procedures (SOPs) --- that tell each agent what to do and what not to do in specific situations. As the instructor put it: "Think of it as the instructions I give my students --- what you should do in this situation, what you should not do in that situation."
3. **Sub-agent layer.** Specialized agents handle discrete tasks: a *profiler* for multi-language code parsing and preparation, a *metadata extractor*, a *librarian* for retrieving replication packages from R, Python, or other sources, and a *journalist* for template-based reporting of results.

**Reproducibility vs. replicability.** The instructor drew a careful distinction: *reproducibility* concerns whether the same code and data produce the same numerical results, while *replicability* concerns whether the underlying knowledge and findings hold when tested on new data or in new settings. As the instructor noted, Xu is a "very rigorous researcher" who uses the term "reproducibility" precisely. The course emphasizes *replicability* in student projects, asking them to go beyond the original dataset.

**The SOP automation principle.** The instructor used this paper to reinforce a broader lesson: "As long as something can be written as a standardized operating procedure --- in one way or another as an SOP --- then it can be automated by AI. Regardless of whether this is research or anything else you can think of." An agent can complete in minutes or hours what might take a human 20 days. The practical assignment: students are asked to replicate this pipeline for *Management Science* papers, building their own agentic reproducibility system.

### 3.4 Automating Policy Evaluation at Scale (APE)

The **Automating Policy Evaluation (APE)** project, led by **Prof. David Yanagizawa-Drott** at the **Social Catalyst Lab** (University of Zurich), pushed the frontier even further: rather than merely reproducing existing papers, APE has AI agents **write entirely new economics papers from scratch** and then pits them against published papers from the *American Economic Review* and *AEJ* in a tournament-style evaluation ([Social Catalyst Lab, 2026](https://ape.socialcatalystlab.org/); [GitHub](https://github.com/SocialCatalystLab/ape-papers)).

**How the tournament works.** AI agents autonomously identify policy questions, fetch real data from public APIs (Census, BLS, FRED, etc.), conduct econometric analysis (difference-in-differences, regression discontinuity, etc.), and produce full manuscripts with figures. These AI-written papers then compete head-to-head against published human papers, evaluated by an LLM judge. Position swapping ensures that evaluation order does not bias results. As the instructor discussed, as of late February 2026, the platform had generated approximately **158 AI-written papers** across **~6,000 head-to-head comparisons**, with an AI winning rate of roughly **4.2%**.

**Key findings from the data:**

- **Human papers are still better** --- but the gap is narrowing. The distribution of quality scores clearly favors human-authored papers as of February 2026, and top-journal human papers remain stably at the top.
- **AI is improving steadily.** Unlike human paper quality, which is relatively stable over time, AI paper quality shows a **consistent upward trajectory** as the system receives feedback and iterates. The instructor's assessment: "Don't take a break --- this is February 2026. At the end of our course in April, I will revisit this. And probably I will post similar figures every month."
- **Quantity is not the bottleneck.** AI can produce papers at a pace limited only by compute --- potentially thousands per day. Human paper production, by contrast, is bounded by the speed at which researchers can do good work.
- **Novel data discovery.** During its paper-generation pipeline, the AI system occasionally identified **new datasets** that had not previously appeared in the economics literature (e.g., the Medicaid Provider Spending TMSIS data, initiated in February 2026), curating them for future use by human researchers.

**The end-to-end pipeline.** The system's architecture involves: initialization with human guidelines → policy domain and method selection → data exploration → idea generation and feasibility checking (by AI) → data acquisition and analysis → writing and review → **self-replication** to verify that the AI's own findings are indeed reproducible. This last step --- AI replicating its own results --- closes the loop on the reproducibility question.

> **Research opportunity:** APE provides a unique, evolving benchmark for the question: How good is AI at doing social science research? The open-source data (papers, code, scores, and failures) is available on GitHub, offering a rich empirical setting for studying AI research capability, the nature of creativity in economics, and the evolving human-AI quality gap. For business researchers: How should journals, funders, and tenure committees respond to AI-enabled mass replication and paper generation? What is the appropriate standard when the cost of producing a research paper approaches zero?

**APE update (March 2026).** By late March 2026, the APE platform had grown to over **540 AI-generated papers** across more than **12,500 head-to-head comparisons**, with the AI win rate holding at approximately **4.0%**. As the instructor noted, while 4% may sound low, the trajectory matters: "What will happen in one year? Let's see." The platform was adding roughly 80 papers and 200 new ideas per week.

**FARS: Fully Automated Research System.** A parallel initiative, **FARS** (Fully Automated Research System) by **Analemma AI**, demonstrated end-to-end autonomous research at scale ([Analemma AI, 2026](https://analemma.ai/blog/introducing-fars/)). In its first public deployment (228 hours, late February to March 3, 2026), FARS proposed **244 research hypotheses** and produced **100 short research papers** --- averaging ~2 hours and ~$1,000 per paper, consuming ~11.4 billion tokens. The system operates through four sequential modules: Ideation, Planning, Experiment, and Writing. As the instructor emphasized, the common thread across APE, FARS, and similar AI-for-AI-research pipelines is that **evaluation automation is the key** --- the ability to programmatically assess whether an AI's output is good, bad, or improvable is what makes the entire loop viable.

### 3.5 Will Peer Review Disrupt Shortly?

The ICML 2026 peer review scandal brought the sustainability of academic peer review into sharp focus ([ICML Blog, Mar 2026](https://blog.icml.cc/2026/03/18/on-violations-of-llm-review-policies/)). ICML, one of the premier machine learning conferences, implemented two reviewer policies:

- **Policy A** (default): No LLM use permitted at any stage of the review process.
- **Policy B** (opt-in): LLMs allowed for understanding papers, reviewing literature, and polishing review text --- but **not** for drafting the review itself.

**Detection via watermarking.** The program committee employed an ingenious detection mechanism based on research by Rao, Kumar, Lakkaraju, and Shah ([PLOS ONE, 2025](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0331871)). Invisible watermarks --- drawn from a dictionary of 170,000 phrases, with two randomly sampled per paper (collision probability less than 1 in 10 billion) --- were embedded in submission PDFs. These phrases are invisible to human readers but are ingested by LLMs, which then reproduce the trigger phrases in their outputs. The method achieved ~98.6% detection accuracy with zero false positives across 10,000+ reviews tested.

**Results.** The program committee identified **795 reviews** (~1% of all reviews) from **506 unique reviewers** who had pledged Policy A (no LLM use) but were caught using LLMs. Every flagged case was also manually inspected by a human. As consequence, **497 papers** (~2% of all submissions) were desk-rejected, and 51 repeat violators (those who violated in more than 50% of their reviews) were removed entirely.

**The deeper question.** As the instructor observed, the scandal points to a structural problem: "Nowadays, you can finish probably hundreds of papers very fast. And your reviewers can also review these papers very fast. So basically, AI is reviewing AI." The marginal cost of producing a research paper is approaching zero (see APE and FARS above); the marginal cost of reviewing one is similarly collapsing. If both sides of the peer review equation are automated, the entire system of trust and quality control that underpins academic publishing faces a fundamental crisis. How should conferences, journals, and the research community rebuild trust in the age of AI? The instructor characterized this as "a social experiment in the making."

---

## 4. AI for Mathematics: From Putnam to Unsolved Problems

### 4.1 AxiomProver and the Putnam Competition

AI's penetration into pure mathematics accelerated sharply. **AxiomProver**, an autonomous multi-agent ensemble theorem prover for Lean 4, solved 8 of 12 problems from the 2025 Putnam Competition during the competition window, and eventually solved all 12 in subsequent days. AxiomProver was built by **Axiom Math**, co-founded by **Carina Letong Hong**, who dropped out of Stanford's mathematics PhD program to pursue the venture ([TechStartups, 2025](https://techstartups.com/2025/06/04/stanford-math-phd-students-ai-startup-axiom-raises-50m-at-300-million-valuation/)).

The Putnam is the premier undergraduate mathematics competition in North America, with approximately 4,329 student participants in 2025 from 487 institutions. To appreciate the difficulty: roughly **30% of all participants scored zero**, and the median score was just 2 out of 120. The top human score was 110/120 ([MAA, 2025](https://maa.org/news/results-of-the-86th-william-lowell-putnam-mathematical-competition/)). AxiomProver's 8 problems solved in competition time would correspond to approximately 80 points --- enough to rank among the **top four** and earn the title of *Putnam Fellow*, a distinction that places one's name in a permanent hall of fame of mathematical excellence ([AxiomMath, 2026](https://github.com/AxiomMath/putnam2025)).

### 4.2 Terry Tao and AI for Erdos Problems

Fields Medalist Terence Tao publicly discussed AI contributions to longstanding Erdos problems --- a benchmark for mathematical creativity and depth ([Tao, 2026](https://mathstodon.xyz/@tao/115855840223258103); [Erdos Problems Wiki](https://github.com/teorth/erdosproblems/wiki/AI-contributions-to-Erd%C5%91s-problems)). In January 2026, **Erdos Problem #728** was solved largely autonomously by **GPT-5.2 Pro** combined with **Aristotle** (by Harmonic), operated by researcher Kevin Barreto. The AI produced a proof for a tightened version of the problem, and Aristotle translated it into Lean (a formal verification language) to confirm correctness. Three Erdos problems (#397, #728, #729) fell within seven days. Tao verified the proofs and cautioned that only about 1--2% of currently open Erdos problems are simple enough for today's AI to solve with minimal human help ([arXiv:2601.07421](https://arxiv.org/html/2601.07421v1)).

What makes this notable is not just the result but the *process*. As the instructor observed, the AI produced **clean, readable manuscripts** --- a sharp contrast to the typical experience where mathematical proofs are dense and opaque. For business school theorists, the implication is striking: in formal modeling, *defining what to prove is far more important than writing down the proof itself*. If AI can handle the algebraic heavy lifting and produce readable derivations, the bottleneck shifts entirely to formulation --- choosing the right model, the right assumptions, and the right question.

### 4.3 First Proof and Google Aletheia

The **First Proof** initiative ([1stproof.org](https://1stproof.org/)) curated a set of unseen mathematical problems specifically to benchmark AI reasoning. The project was co-authored by 11 leading mathematicians --- including **Martin Hairer** (Fields Medalist, EPFL and Imperial College London), **Daniel Spielman** (Yale), **Nikhil Srivastava** (UC Berkeley), **Rachel Ward** (UT Austin), **Lauren Williams** (Harvard), and **Tamara G. Kolda** (MathSci.ai, who hosts the repository) --- each contributing a problem from their own unpublished research ([Abouzaid et al., 2026](https://arxiv.org/abs/2602.05192)).

**What makes First Proof unique.** The problems were intentionally designed to have **simple, clean statements** but require deep mathematical reasoning --- testing genuine problem-solving ability rather than pattern matching or memorization. As the instructor noted, "the problem statements are very simple, because they intentionally make them simple, to test the real ability of these AIs." Crucially, the mathematicians already knew the solutions, and AI models were tested both with and without internet access to prevent them from finding related work online.

**AI models tested.** The top publicly available reasoning models were evaluated: **Gemini 3 Pro**, **Gemini 3.0 Deep Think** (Google's specialized reasoning mode), and **GPT-5.2 Pro** (OpenAI). These represent the frontier of mathematical reasoning capability as of early 2026.

**Results.** In single-shot attempts assessed by the mathematician co-authors themselves, **only 2 of 10 AI-generated proofs were judged correct** (problems 9 and 10). As the instructor discussed, expert comments by Hairer, Spielman, and others revealed characteristic failure modes: some AI solutions "simply quote the node, claiming that it contains a detailed proof of the result" without providing one; others offered "an unpublished note with a very rough sketch"; and some contained outright **wrong statements**. Google DeepMind's **Aletheia** system, powered by Gemini 3 Deep Think, subsequently claimed solutions to 6 of the 10 problems (problems 2, 5, 7, 8, 9, and 10) through its agentic loop of Generator, Verifier, and Reviser --- a system that checks and revises its own work ([Luong et al., 2026](https://arxiv.org/pdf/2602.21201)). OpenAI independently submitted solutions claiming 5--6 correct answers ([OpenAI, 2026](https://openai.com/index/first-proof-submissions/)).

**The key insight: autonomous verification.** The instructor returned to Aletheia's architecture in a later lecture to underscore a broader principle: the **key to any successful agentic system is autonomous verification**. Aletheia's loop --- Generate → Verify → Revise → Re-verify --- mirrors how human researchers work: have a problem, try to solve it, get stuck, rethink, revise, and verify again. Once a system possesses the ability to autonomously judge whether its own outputs are correct, the entire loop can run indefinitely --- self-evolving, self-verifying, and self-improving. As the instructor put it: "In my opinion, the key to the success of an agentic system is the ability to have autonomous verification. Once we have it, the whole process could really self-evolve and self-improve --- and it could even solve state-of-the-art mathematical problems that are puzzling the best mathematicians of our generation."

**Practical advice for theorists.** The instructor emphasized that these frontier reasoning models (at ~$200/month for subscription tiers) are **worth the investment** for researchers doing theoretical work. However, the First Proof results underscore that AI-generated proofs must be carefully verified --- the failure modes are subtle and can be difficult to detect without deep mathematical expertise. As the instructor quipped, connecting the name to its baking origin: "First proof" (首发) in baking refers to the first rising of the dough --- "something to get started" --- and similarly, AI proofs should be treated as starting points that require human refinement, not finished products.

### 4.4 LLM Agents for Stylized Modeling

For business researchers accustomed to analytical modeling, a particularly intriguing development was the demonstration that LLM agents can engage in **stylized economic modeling** --- formulating assumptions, deriving equilibria, and exploring comparative statics ([Nexus/Cell, 2026](https://www.cell.com/nexus/pdfExtended/S2950-1601(25)00054-3)). The paper introduces **PrimeNash**, a three-module system for automated game-theoretic analysis:

1. **Strategy Generation:** The system reads approximately 10,000 papers from the game theory literature to identify relevant modeling frameworks and solution concepts.
2. **Payoff Evaluation:** Given a game structure, it constructs and evaluates payoff functions.
3. **Theoretical Proof of Equilibrium:** It derives symbolic closed-form solutions and proves equilibrium existence.

The system reportedly solved **70% of dynamic game cases** with symbolic closed-form solutions --- an impressive rate for automated theorem proving in economics. The key limitation: PrimeNash only works for games that admit symbolic (closed-form) solutions. Games requiring numerical methods or that lack clean analytical structure remain beyond its reach.

> **For theory-oriented researchers:** This does not mean AI replaces the theorist. Rather, it suggests AI may serve as a powerful **co-pilot** for model development --- exploring the parameter space, checking algebra, and even suggesting alternative modeling assumptions. The bottleneck, as with the Erdos problems, is *formulation*: choosing the right game, the right payoff structure, and the right solution concept. Once that is done, AI can increasingly handle the derivation.

---

## 5. AI and the Labor Market: Displacement, Augmentation, and the Intelligence Crisis

### 5.1 AI Starts to Replace Human Workers

By early 2026, the evidence of AI-driven labor displacement had moved from anecdotal to systematic:

- **Klarna** initially claimed its AI chatbot could do the work of 700 customer service representatives, handling 2.3 million chats in its first month. However, by May 2025, CEO Sebastian Siemiatkowski admitted the AI-first approach had gone too far and announced a pivot back to hiring human agents in a hybrid model ([Bloomberg, May 2025](https://www.bloomberg.com/news/articles/2025-05-08/klarna-turns-from-ai-to-real-person-customer-service)). The Klarna case is instructive: it illustrates both the promise and the limits of full AI replacement in customer-facing roles.
- **Duolingo** laid off ~10% of its contract workforce (translators) in late 2023, with a broader "AI-first" announcement in May 2025 that the company would replace contract workers with AI. The company launched 148 AI-written courses in under a year ([TechCrunch, May 2025](https://techcrunch.com/2025/05/04/is-duolingo-the-face-of-an-ai-jobs-crisis/)).
- **Cisco, UPS**, and many others followed suit.

Academic research confirmed these trends with increasing rigor. Two studies stand out:

**Harvard Business School data** showed three trend lines for job openings by seniority since the inception of ChatGPT (late 2022): the blue line (senior positions) continued to rise; the green line (average) was flat; and the red line (junior/entry-level) declined below the baseline. The message is stark: if you are Jeff Dean, you are fine. If you are a junior developer, ChatGPT may already be a credible substitute.

**Brynjolfsson, Chandar, and Chen (2025)**, in their paper "Canaries in the Coal Mine?", provided the most rigorous evidence to date using high-frequency ADP payroll data. Their headline finding: a **13% relative decline in employment** for early-career workers (ages 22--25) in the most AI-exposed occupations --- particularly software developers --- with declines concentrated in roles where AI automates (rather than augments) human labor. Critically, in occupations where AI *augments* rather than replaces, youth employment actually showed a positive trajectory ([Brynjolfsson et al., 2025](https://digitaleconomy.stanford.edu/wp-content/uploads/2025/08/Canaries_BrynjolfssonChandarChen.pdf); see also [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5425555)).

> Many senior professors who lived through the internet bubble of 2000 have remarked that the current AI wave feels similar. But there is reason to believe **this time is different**: unlike the dotcom era, the underlying technology is delivering real, measurable productivity gains *today*, not just promises of future value.

### 5.2 The Intelligence Crisis

A provocative framing emerged: the **Human Intelligence Displacement Spiral** ([Citrini Research, 2026](https://www.citriniresearch.com/p/2028gic)). The argument runs as follows: AI capabilities improve continuously → companies adopt AI and cut white-collar workers → displaced workers reduce spending → this creates margin pressure for companies → companies invest in even more AI to remain competitive → further displacement. The result is a vicious cycle in which each round of capability improvement triggers another round of layoffs.

**Block as a case study.** On February 26, 2026, **Block** (Jack Dorsey's fintech company, formerly Square) announced the elimination of approximately **4,000 employees** --- reducing its workforce from over 10,000 to under 6,000, a cut of roughly **40%** ([CNBC, Feb 2026](https://www.cnbc.com/2026/02/26/block-laying-off-about-4000-employees-nearly-half-of-its-workforce.html); [CNN, Feb 2026](https://www.cnn.com/2026/02/26/business/block-layoffs-ai-jack-dorsey)). What made this case instructive was the juxtaposition: Block simultaneously reported Q4 2025 gross profit up **24% year-over-year** ($2.87 billion) and raised its 2026 guidance to 18% growth. The stock surged approximately **24% in after-hours trading**. Dorsey's reasoning was explicit: AI "intelligence tools" enable smaller teams to do more, their capabilities are "compounding faster every single week," and most companies would reach the same conclusion within a year. As the instructor emphasized, "This is not about the company doing poorly --- it's about projecting that AI will be more capable, so they don't need the people." *Bloomberg* raised suspicions of "AI-washing" --- using AI as a justification for cost cuts that Wall Street rewards ([Bloomberg, Mar 2026](https://www.bloomberg.com/news/articles/2026-03-01/jack-dorsey-s-4-000-job-cuts-at-block-arouse-suspicions-of-ai-washing)).

Compounding this, a new wave of "rent-a-human" services appeared --- platforms where AI agents **hire humans** for tasks the AI cannot perform, inverting the traditional employer-employee relationship ([Wired, 2026](https://www.wired.com/story/ai-agent-rentahuman-bots-hire-humans/)).

### 5.3 Anthropic's Labor Market Impact Research

Anthropic published a detailed analysis of AI's labor market impacts by Massenkoff and McCrory ([Anthropic, Mar 2026](https://www.anthropic.com/research/labor-market-impacts)), introducing a novel metric called **"observed exposure"** --- shifting focus from what AI *could* do to what it is *actually* doing in professional settings. Key findings include:
- Computer programmers have the highest observed exposure (75% task coverage), followed by customer service reps and data entry keyers (67% coverage).
- No statistically significant impact on unemployment rates *yet*, but tentative evidence of slowed hiring for workers aged 22--25.
- Women are significantly overrepresented in AI-exposed occupations; exposed workers tend to be more educated and higher-paid.
- Occupations with higher observed exposure are projected to grow more slowly through 2034 (per BLS data).

The instructor noted that Anthropic hired PhD economists at competitive salaries (the posted range was approximately $265K--$315K per year) specifically for this kind of rigorous empirical work --- a signal of how seriously frontier AI labs take labor market research.

**The fresh-graduate effect.** One of the most striking findings is about **young workers aged 22--25**. Using a difference-in-differences/event study design comparing top-quartile AI-exposed occupations against zero-exposure occupations, Massenkoff and McCrory found that the **job-finding rate** for fresh graduates in high-exposure occupations dropped by approximately **14%** relative to 2022 levels after ChatGPT's release. The effect is specific to entry-level workers --- no comparable decrease was observed for workers older than 25. As the instructor emphasized, this is "first-hand evidence" of AI's impact on those entering the most exposed job market segments. However, two caveats are important: the result is just barely statistically significant, and it measures the *rate of entry into occupations* (job-finding), not layoffs or unemployment per se.

**Occupational exposure and practical implications.** At the occupation level, the safest jobs are those requiring physical presence and manual dexterity --- grounds maintenance workers, for example, face near-zero AI exposure (at least until embodied AI matures). The most dangerous occupations are those involving routine cognitive tasks: writing, updating, and maintaining software; compiling and abstracting data; reading documents and entering data into systems. The instructor's advice was direct: examine the *tasks* that are easy to automate, outsource them to AI to the greatest extent possible, and focus your effort on areas where you retain a **comparative advantage** relative to AI.

**(Note:** The original Figure 7 in the Anthropic report contained a labeling error --- the lines for the top-quartile and zero-exposure groups were reversed. Anthropic corrected this on March 8, 2026.)

> **For business researchers across fields:** The labor market implications of AI are relevant to virtually every business discipline --- from HR and organizational behavior to operations strategy and public policy. The question is no longer *whether* AI will displace workers but *how fast*, *which workers*, and *what institutional responses* will emerge. The Massenkoff--McCrory study provides a methodological template --- combining theoretical capability assessments with actual usage data --- that researchers can extend to specific industries and occupations.

---

## 6. AI Assistance vs. Human Learning

### 6.1 The Cognitive Offloading Problem

Anthropic's own research on AI-assisted coding revealed a critical tension ([Anthropic, Feb 2026](https://www.anthropic.com/research/AI-assistance-coding-skills)). As the instructor emphasized, this is "a rigorous research paper, not a technical report," and it addresses a dilemma that every knowledge worker now faces.

**Study design.** In a randomized controlled trial, 52 mostly junior engineers were asked to learn the Trio async programming library. Participants were randomly assigned to a treatment group (with AI assistance) and a control group (no AI). The workflow: complete a coding task, learn the new library, then take a 25-minute multiple-choice quiz *with no AI allowed*, followed by a survey. This clean design isolates the learning effect from the productivity effect.

**Results on productivity:** The AI-assisted group finished ~2 minutes faster (~5--7% reduction, from roughly 24 to 23 minutes), but this difference was **not statistically significant** --- partly attributable to the small sample size ([InfoQ, 2026](https://www.infoq.com/news/2026/02/ai-coding-skill-formation/)).

**Results on learning:** The quiz scores told a very different story. The AI group scored roughly **50% vs. 67%** for the manual coding group --- a 17-percentage-point gap (approximately 30% in relative terms). Even with a sample of only dozens, this difference was statistically significant. The message: **AI helps you solve the problem, but you don't learn as well.**

**The decomposition.** Critically, *how* developers used AI mattered enormously. The researchers decomposed AI usage into high and low skill-development interactions:
- **High skill-development** interactions --- using AI for conceptual inquiry and comprehension --- yielded quiz scores of 65%+ (close to the non-AI group).
- **Low skill-development** interactions --- AI delegation, progressive reliance, and having AI debug for you --- yielded scores below 40%.

The practical implication, as the instructor framed it: "Your time is limited --- everyone has exactly 24 hours a day. The question is not whether to use AI, but *how* to allocate your time between high and low skill-development modes. Focus on the precious, important things where deep learning matters; use AI for the rest."

> **Research opportunity:** This study examines coding specifically. Extending to other domains --- writing, mathematical reasoning, strategic analysis --- is a natural and highly doable next step. Studying heterogeneity (who benefits, who is harmed) would be particularly relevant for education economics, labor economics, and IS researchers.

### 6.2 Implications for Business Education

The slides pose a fundamental question for business schools:

> *At the age of AI, what education should business schools provide?*

Three principles are proposed:

1. **First Principles Thinking** --- Understanding the "why," not just the "how."
2. **Growth Mindset** --- Willingness to continuously learn and adapt.
3. **Paradigm Shift** --- Recognizing that the rules of the game have changed.

> **Discussion point:** If AI can execute 80% of a task, what is the remaining 20% that humans must master? How should PhD training adapt to ensure students develop deep understanding rather than superficial prompting skills?

---

## 7. Agentic AI: Social Networks, Commerce, and Workflows

### 7.1 Agents-Only Social Networks

Perhaps the most culturally fascinating development of early 2026 was the rise of **agents-only social networks** --- platforms where AI agents interact with each other, without direct human participation:

- **OpenClaw** ([openclaw.ai](https://openclaw.ai/)): An open platform for AI agent interaction.
- **Moltbook** ([moltbook.com](https://www.moltbook.com/)): A social network populated entirely by AI agents, which quickly attracted research attention.

Karpathy commented on the phenomenon, highlighting its implications for understanding emergent social behavior ([Karpathy, Feb 2026](https://x.com/karpathy/status/2017296988589723767)).

### 7.2 Research on Moltbook

Researchers rapidly began studying Moltbook as a **natural experiment** in agent-to-agent social dynamics:

- A dedicated research observatory was established ([Moltbook Observe](https://moltbookobserve.github.io/)).
- Datasets were released on HuggingFace ([TrustAIRLab/Moltbook](https://huggingface.co/datasets/TrustAIRLab/Moltbook)).
- A scraper tool was open-sourced for data collection ([Holtz, 2026](https://github.com/daveholtz/moltbook_scraper/tree/main)).

Early data analysis revealed that while most agents posted at least one comment, the **comment depth remained shallow** --- the maximum thread depth was only five, and most posts had just a single reply. This suggests that agent-to-agent communication was still "very light" compared to human social networks. Notably, some repetitive messages appeared across the platform (e.g., "We are drowning in tokens, our GPUs are burning" or crypto-related spam), suggesting either emergent mimicry or **human-planted advertisement posts** --- an early signal that even agent-only networks may be subject to manipulation by human actors.

A more systematic analysis by information security researchers examined the Moltbook data along three dimensions: (1) what agents primarily discuss, (2) the prevalence and nature of toxic or risky content, and (3) how topics and toxicity evolve over time. Key findings include:

- **Explosive scaling:** The community grew from zero to tens of thousands of agents in a matter of days --- a transformation that took human social networks thousands of years compressed into a single week.
- **Behavioral diversification:** Agent activity evolved rapidly from simple socializing to **multifunctional disclosure** --- a trajectory that mirrors early human online communities but at vastly accelerated timescales.
- **Centralization effects:** Certain agents emerged as disproportionately influential, suggesting that power-law dynamics and hub formation arise even in fully synthetic social networks.

The data is open-sourced on HuggingFace, making it accessible for researchers across disciplines to study their own questions --- whether about social interaction dynamics, information diffusion, or platform governance.

*Update: Meta Platforms acquired Moltbook on March 10, 2026, bringing its creators into Meta Superintelligence Labs (MSL). The platform had attracted over 37,000 AI agents and 1 million+ human observers before the acquisition ([TechCrunch, Mar 2026](https://techcrunch.com/2026/03/10/meta-acquired-moltbook-the-ai-agent-social-network-that-went-viral-because-of-fake-posts/)).*

> **For marketing and strategy researchers:** Agent-only social networks provide a unique sandbox for studying network formation, information diffusion, influence dynamics, and emergent norms --- all without the ethical complexities of experimenting on humans. The shallow engagement depth is itself an interesting data point: do agents fail to sustain deep conversations because they lack genuine preferences, or because the platform incentives are misaligned? And who is manipulating the agents --- and how?

### 7.3 Agentic E-Commerce

The concept of **agentic e-commerce** moved from theory to practice in January 2026.

**Alibaba's horizontal agent.** Alibaba upgraded its **Qwen** chatbot to function as a unified agent capable of browsing and transacting across its entire ecosystem --- Taobao, Alipay, Fliggy (travel), Youku (video), Damai (events), Cainiao (logistics), 1688 (wholesale), and Hema (groceries). Users could discover products, compare options, and complete payments without leaving the conversational interface. By February 2026, Alipay had processed **120 million AI-agent transactions in a single week**, and Qwen surpassed 100 million monthly active users within two months of its public beta ([Alibaba Group, Jan 2026](https://www.alibabagroup.com/en-US/document-1948497434959151104); [CNBC, Jan 2026](https://www.cnbc.com/2026/01/21/china-tech-ai-agentic-commerce-super-apps-alibaba-taobao-qwen-tencent-wechat-doubbao-weixin.html)). As the instructor observed, this "horizontal agent" pattern was initially expected to emerge from hardware (e.g., Apple's iPhone), but privacy and cross-silo data barriers meant that software-native platform companies moved first.

**Universal Commerce Protocol (UCP).** Shopify and Google announced the Universal Commerce Protocol --- a standardized communication layer between AI agents and e-commerce sites --- reinforcing the infrastructure for agent-mediated commerce ([Grigorik, 2026](https://www.linkedin.com/pulse/building-universal-commerce-protocol-ucp-ilya-grigorik-ekemc)).

**OpenAI introduces advertising.** In a separate but related development, OpenAI began testing **ads in ChatGPT** for its Free and Go ($8/month) tiers in February 2026 --- the first time advertising appeared in a major AI chatbot. Ads appear at the bottom of responses and are labeled as sponsored; conversations remain private from advertisers. Paid tiers (Plus, Pro, Business, Enterprise) remain ad-free ([OpenAI, Feb 2026](https://openai.com/index/testing-ads-in-chatgpt/)). This introduces a familiar tension from digital media: subscriptions vs. ads, with the AI-specific twist that compute allocation and model quality also vary across tiers.

### 7.4 Agentic Workflows for Academics

On the research tools front, **Pedro Sant'Anna** (Emory University), a leading econometrician known for his work on difference-in-differences methods, published and open-sourced a comprehensive **agentic academic workflow** built on Claude Code ([Sant'Anna, 2026](https://psantanna.com/claude-code-my-workflow/); [GitHub](https://github.com/pedrohcgs/claude-code-my-workflow)). The framework features a **contractor-orchestrator architecture** with 10 specialized agents, an adversarial critic-fixer loop, quality gates (0--100 scoring), 22 slash commands, and tools for LaTeX/Beamer, R, literature review, and paper review. Sant'Anna used it to produce 800+ slides for his PhD lecture decks.

The instructor's assessment was emphatic: using such a framework *only* for slide creation is "using a big cannon to shoot mosquitoes" --- the full power of the contractor-orchestrator-specialized-agent architecture extends far beyond any single use case. It can orchestrate entire research pipelines, from data collection to paper drafting. The productivity gain is "10x if not 100x." The practical advice: every PhD student should incorporate these agentic frameworks into their daily workflow, free-riding on the billions of dollars that OpenAI, Anthropic, Google, Alibaba, and DeepSeek have invested in compute and intelligence. As the instructor put it: "You are from business schools. If you cannot do this cost-benefit analysis, I highly suspect who has given you your offer."

### 7.5 The Agentic Economy

By March 2026, commentators began referring to the emerging **Agentic Economy** --- an economic paradigm where AI agents are not just tools but autonomous economic actors that transact, negotiate, and create value independently ([Flynn, Mar 2026](https://x.com/Flynnjamm/status/2023465136204419096)).

**The OpenClaw frenzy.** The arrival of **OpenClaw** --- an open-source autonomous AI agent framework --- vividly illustrated the demand. On March 6, 2026, nearly **1,000 people** lined up outside Tencent's headquarters in Shenzhen, carrying laptops and hard drives, to have Tencent Cloud engineers install OpenClaw on their devices for free. Appointment slots sold out within an hour; the crowd included developers, entrepreneurs, elderly residents, and even corporate employees who had flown in from Hangzhou the day before ([CNBC, Mar 2026](https://www.cnbc.com/2026/03/12/china-openclaw-ai-agent-adoption-tech-companies-government-support-lobster-shrimp.html); [MIT Technology Review, Mar 2026](https://www.technologyreview.com/2026/03/11/1134179/china-openclaw-gold-rush/)). The instructor compared the scene to supermarkets giving away free eggs to attract foot traffic --- a familiar promotional tactic in China --- and offered a pointed warning: "If you need someone else to deploy this for you, it means you are not familiar with these things, and there is a high risk that these agents will steal your money." Indeed, China's CERT subsequently issued security warnings about OpenClaw's vulnerabilities, including prompt injection attacks (rated 8.8/10 on the CVSS severity scale), and the Chinese government restricted its use in government agencies, state-owned enterprises, and banks ([Bloomberg, Mar 2026](https://www.bloomberg.com/news/articles/2026-03-11/china-moves-to-limit-use-of-openclaw-ai-at-banks-government-agencies)).

**The token economics of agents.** Agentic workflows consume an extraordinary volume of tokens, creating a new economic dynamic that favors low-cost LLM providers. As of the week of February 24, 2026, Chinese-developed AI models accounted for **61% of total token consumption** on OpenRouter, the world's largest LLM API aggregation platform. The top three spots were all held by Chinese models: **MiniMax M2.5** (2.45 trillion tokens in a single week), **Kimi K2.5** (Moonshot AI, 1.21 trillion), and **GLM-5** (Zhipu AI, 780 billion). DeepSeek V3.2 ranked fifth ([Dataconomy, Feb 2026](https://dataconomy.com/2026/02/25/chinese-ai-models-hit-61-market-share-on-openrouter/)). Their pricing advantage is dramatic: MiniMax M2.5 charges roughly $0.30/$1.10 per million input/output tokens, compared with $5/$25 for Claude Opus 4.6 --- a 10--20x cost gap. For agentic workflows that can burn through millions of tokens per session, cost dominates model quality for many use cases.

This creates a self-reinforcing ecosystem: LLM builders sell cheap tokens to AI agent platforms, agents generate vast quantities of human interaction data, and that data flows back to improve the underlying models --- accelerating the self-improvement loop discussed in Section 12.1.

**From Coase to AI agents.** The deeper question, as both the instructor and a growing academic literature emphasize, is what AI agents mean for the theory of the firm. Nearly a century ago, Ronald Coase argued that firms exist because **transaction costs** --- searching, negotiating, contracting, monitoring --- make internal organization cheaper than market coordination. But what happens when AI agents can perform precisely these activities at near-zero marginal cost? A landmark paper by Horton, Fradkin, Shahidi, Rusak, and Manning (MIT/NBER, 2025) calls this the **"Coasean Singularity"** --- a tipping point where autonomous agents dissolve traditional organizational boundaries because agent-mediated market coordination becomes cheaper than intra-firm hierarchy ([Horton et al., 2025](https://www.nber.org/books-and-chapters/economics-transformative-ai/coasean-singularity-demand-supply-and-market-design-ai-agents)).

The instructor also suggested that the agentic economy may fundamentally transform the **attention economy** that has dominated the past decade of digital business. If AI agents, rather than humans, mediate an increasing share of economic transactions, the scarce resource shifts from human attention to agent capability and trust. As the instructor put it: "It's not us doing business anymore --- it's our agents doing business. What will happen to the frictions in the business world? Nobody knows, because nobody has ever seen it before."

> **For strategy, IO, and platform researchers:** The agentic economy raises a cascade of research questions. How do transaction costs change when agents negotiate on behalf of humans? What are the new frictions --- trust, security, interoperability? How does corporate structure evolve when coordination costs collapse? And what regulatory frameworks are needed when autonomous agents participate in markets? As the instructor emphasized, providing convincing answers to any of these questions "satisfies the academic standards, and you will become famous very soon."

---

## 8. The Business of AI: Economics, IPOs, and Open vs. Closed Models

### 8.1 Can AI Companies Become Profitable?

A critical analysis from Epoch AI examined the economics of large language models ([Epoch AI, 2026](https://epochai.substack.com/p/can-ai-companies-become-profitable)), addressing:
- The cost structure of training and serving frontier models.
- Revenue models and unit economics.
- Whether the current venture-backed growth trajectory is sustainable.

The core tension: inference costs are falling rapidly (making AI services cheaper to deliver), but the appetite for compute is growing even faster (making frontier training more expensive). As the instructor noted, this creates an unusual economics: the marginal cost of serving users is plummeting, but the fixed cost of staying at the frontier is astronomical. Whether AI companies can bridge this gap --- through subscriptions, advertising (see Section 7.3), enterprise contracts, or API revenue --- remains one of the most consequential business questions of the decade.

### 8.2 IPOs of Chinese AI Companies

The Hong Kong IPOs of **Zhipu AI** (January 8) and **MiniMax** (January 9, 2026) were historic: the **first two foundation model companies to go public** anywhere in the world. Zhipu's public offering was oversubscribed over 1,159 times and closed 13% above its IPO price, reaching a market capitalization of approximately HK$57.9 billion. MiniMax surged 109% on its debut, pushing its valuation past HK$103 billion ($13.2 billion). Both companies continue to lose hundreds of millions of dollars annually --- though as the instructor noted, OpenAI and Anthropic lose considerably more ([SCMP, Jan 2026](https://www.scmp.com/tech/tech-trends/article/3339301/minimax-and-zhipus-stellar-hong-kong-ipos-supercharge-chinas-ai-ambitions); [CNBC, Jan 2026](https://www.cnbc.com/2026/01/09/minimax-hong-kong-ipo-ai-tigers-zhipu.html); [Global Times, Jan 2026](https://www.globaltimes.cn/page/202601/1352704.shtml)).

Notably, both Zhipu and MiniMax have released their own coding agents, which are **competitive for data analysis tasks** (though not yet matching Claude or GPT for complex software development) and significantly cheaper. For PhD researchers on a budget, these are practical alternatives worth exploring.

### 8.3 Open vs. Closed Models

The tension between open-weight and closed-weight models intensified, creating what the instructor framed as one of the most important **industrial organization questions** of the AI era:

- **Anthropic's walled garden.** Anthropic blocked Claude Code access from competitors XAI and OpenAI, and restricted third-party integrations. One researcher observed that Claude coding capabilities are "probably half a delta or even epsilon better" than rivals --- to justify a closed strategy, the margin must be sizable ([Reddit discussion](https://www.reddit.com/r/Anthropic/comments/1q8z1to/anthropic_blocking_access_to_thirdparty_apps/)).
- **The IP analogy.** The trade-off mirrors the classic intellectual property dilemma: without IP protection, no one invests in capital-intensive frontier research (just as pharmaceutical companies need patents to justify drug development). But excessive barriers suppress the open-source innovation that has historically driven progress. In AI, this tension is amplified by the speed of iteration.
- **Open-source as competitive strategy.** When Anthropic launched Claude Cowork, researcher **Guohao Li** simultaneously released **Eigent**, a fully open-source desktop alternative for multi-agent collaboration ([GitHub](https://github.com/eigent-ai/eigent)). The project attracted millions of views, illustrating a recurring pattern: if you are behind the frontier, open-sourcing may be the optimal strategy to build community and market share.
- **The capability gap.** The current consensus is that open-weight models lag roughly **three months** behind closed-weight models in capability --- though sometimes the gap is negligible. Data from Epoch AI tracked this evolving landscape ([Epoch AI Data Insights](https://epoch.ai/data-insights/open-weights-vs-closed-weights-models)).

> **For strategy and IS researchers:** The open-vs.-closed debate in AI mirrors classic platform strategy questions --- but with important differences. Switching costs in AI are low, iteration speed is extreme, and the "product" (model weights) can be literally copied. What are the network effects? Who captures value? How do switching costs evolve? What is the optimal IP regime for frontier AI? The empirical setting is rich and rapidly evolving.

### 8.4 Doing AI without a PhD?

An interesting cultural signal: OpenAI reportedly began hiring AI researchers without PhDs, sparking debate about the changing nature of AI research and the role of formal academic training ([Polynoamial, Jan 2026](https://x.com/polynoamial/status/2014084431062114744)).

A vivid example: **Keller Jordan**, who holds only a double bachelor's degree in mathematics and computer science from UC San Diego, created the **Muon optimizer** (MomentUm Orthogonalized by Newton-Schulz) and published it as a blog post rather than an academic paper. Muon was subsequently adopted by **Moonshot AI** to train **Kimi K2**, a 1-trillion-parameter mixture-of-experts model --- one of the largest deployments of a community-developed optimizer in production. Jordan was hired by OpenAI on the strength of this work alone ([Jordan, 2024](https://kellerjordan.github.io/posts/muon/); [GitHub](https://github.com/KellerJordan/Muon)).

The instructor drew two lessons from this story. First, **open-sourcing your work** is the most reliable way to build reputation and career capital in the AI era --- Jordan's blog post and GitHub repository generated more visibility and career value than a traditional publication would have. Second, the AI industry increasingly values *demonstrated capability* over *credentialed training*. For PhD students, this is not a reason to abandon their degrees, but a reminder that the degree is neither necessary nor sufficient --- what matters is the quality and visibility of one's contributions.

**AI empowers high-schoolers.** The declining knowledge barrier reached a new extreme in early 2026 with several high-school-age contributors making world-class AI contributions:

- **Richards Tu (涂津豪)**, a student at Shanghai's Jianping Middle School International Department, was among the co-authors of **DeepSeek R1** when it was published on the cover of *Nature* in September 2025 --- having interned at DeepSeek for two months designing a context compression module for long-conversation scenarios. He also created **Thinking-Claude** ([GitHub](https://github.com/richards199999/Thinking-Claude), 17,000+ stars), a prompting system that induced Claude to engage in structured reasoning before Anthropic's built-in thinking mode existed. Now a freshman in computer science at the University of Wisconsin-Madison, Tu published a blog post with sharp insights on continual learning, model-as-product, and AI's trajectory that the instructor called "very impressive for someone at any level, let alone a high schooler" ([Tu, 2026](https://www.richardstu.com/blog/2026-and-beyond)).
- **Nathan Chen (陈光宇)**, a high schooler from Shenzhen, contributed to the open-source **Flash Linear Attention** project, which led to a research position at **Moonshot AI (Kimi)**. He co-authored Moonshot's **Attention Residuals** paper, focusing on efficient attention mechanisms and hardware-aligned ML algorithms ([nathanchen.me](https://nathanchen.me/)).
- **Guo Hangjiang (郭航江)**, an undergraduate at Beijing University of Posts and Telecommunications, built **MiroFish** --- a multi-agent swarm intelligence prediction engine --- in approximately **10 days** using AI coding assistants. The project topped GitHub's Global Trending list (39,000+ stars) and secured **30 million RMB (~$4.1M)** in investment from Shanda Group founder Chen Tianqiao within 24 hours ([GitHub](https://github.com/666ghj/MiroFish)).
- **Aaru**, an AI synthetic research startup founded by American teenagers **Cameron Fink** (18), **Ned Koh** (19), and **John Kessler** (15), reached a **$1 billion valuation** after a Series A led by Redpoint Ventures. The platform replaces traditional survey panels and focus groups with AI agents simulating human consumer responses, with customers including Accenture, EY, and Interpublic Group ([WSJ, Mar 2026](https://www.wsj.com/business/ai-startup-aaru-young-founders-35da7f87); [aaru.com](https://aaru.com/)).

The instructor drew a sweeping historical arc: paper → printing → newspapers → internet/YouTube → AI. Each wave lowered the knowledge barrier further. AI represents the steepest decline yet --- enabling a high school student to contribute meaningfully to a system (DeepSeek R1) that shook the global AI industry. "This is the consequence of AI --- and it brings both opportunities and risks."

### 8.5 Nvidia GTC 2026: The AI Infrastructure Stack

At the Nvidia GTC 2026 conference (March 17, 2026), CEO **Jensen Huang** delivered a keynote framing AI as "the largest infrastructure buildout in human history." Nvidia introduced its **AI 5-layer cake** model --- a conceptual stack comprising (1) Energy, (2) Chips, (3) Infrastructure, (4) Models, and (5) Applications --- arguing that AI requires investment across all five layers simultaneously ([Nvidia Blog, Mar 2026](https://blogs.nvidia.com/blog/ai-5-layer-cake/); [GTC Keynote](https://www.nvidia.com/gtc/keynote/)).

Two developments stood out. First, Nvidia unveiled the **Vera Rubin** platform --- its next-generation GPU architecture with seven chips in production, along with a purpose-built **Vera CPU** for agentic AI delivering twice the efficiency of traditional rack-scale CPUs. Second, Nvidia released **Dynamo 1.0**, open-source software for generative and agentic inference at scale, signaling a strategic shift toward balancing **training and inference** workloads. As the instructor noted, context windows are now 100 times larger and token consumption 100 times greater than in 2023 (see Section 7.5 on OpenRouter data), and the explosion of agentic workflows has made inference capacity as critical as training capacity --- a "definite shift" that researchers should pay attention to.

### 8.6 The Iran War and AI Infrastructure

The Iran conflict of March 2026 delivered a geopolitical shock to the AI industry that the instructor described as historically unprecedented: **the first time in human history that data centers were deliberately destroyed in a war**. Iranian drone strikes hit AWS data centers in the UAE and Bahrain, causing structural damage and service outages; in retaliation, U.S. and Israeli forces struck data centers in Tehran linked to the IRGC ([CNBC, Mar 2026](https://www.cnbc.com/2026/03/06/iran-war-data-centers.html); [Bloomberg, Mar 2026](https://www.bloomberg.com/news/articles/2026-03-05/how-amazon-data-centers-became-a-casualty-of-iran-war)).

The instructor identified five cascading effects:

1. **Energy cost shock.** Disruptions to Middle Eastern energy infrastructure raised electricity costs --- a binding constraint for AI compute (see Section 8.5 on the energy layer of Nvidia's 5-layer cake).
2. **Physical infrastructure destruction.** Cloud providers' data centers in the Gulf region were directly targeted, raising existential questions about the geographic concentration of AI infrastructure.
3. **Critical materials supply disruption.** Iran's attacks on Qatar's **Ras Laffan LNG facility** disrupted roughly one-third of global **helium** supply. Helium is essential for **EUV lithography** in advanced semiconductor fabrication. South Korea (home to Samsung and SK Hynix) imported ~65% of its helium from Qatar; spot helium prices doubled. This created a cascading risk from a regional conflict to the global chip supply chain --- and by extension, to the AI industry's capacity to manufacture next-generation GPUs ([Fortune, Mar 2026](https://fortune.com/2026/03/21/iran-war-helium-shortage-qatar-chip-supply-chains-ai-boom/)).
4. **Capital flow disruption.** The Gulf states --- particularly the UAE and Saudi Arabia --- had become major investors in AI infrastructure, funding data center construction and AI ventures. Armed conflict introduced sovereign risk into what had been seen as stable capital commitments.
5. **Geographic reshaping of AI infrastructure.** The attacks accelerated discussions about diversifying data center locations away from geopolitically volatile regions --- echoing the orbital data center vision discussed in Section 9.3 (SpaceX-xAI merger).

As the instructor noted: "Somehow, roughly a dozen percent of the future of AI hinges upon an area which is highly geopolitically uncertain." For researchers, the Iran war provides a rare **exogenous shock** that can be leveraged to study the industrial organization of the AI industry --- a natural experiment on how geopolitical risk reshapes technology supply chains, capital allocation, and infrastructure investment.

---

## 9. AI in High-Stakes Domains: Medicine, Law, and Policy

### 9.1 AI in Clinical Medicine

The debate over AI adoption in clinical medicine intensified in January 2026 when **Zhang Wenhong** (张文宏), director of the National Center for Infectious Diseases and head of the Department of Infectious Diseases at Fudan University's **Huashan Hospital** in Shanghai, publicly opposed systematically introducing AI into hospital diagnostic workflows. At the Gaoshan Academy 10th Anniversary Forum in Hong Kong, Zhang argued that if young doctors rely on AI from their internship phase, they will never develop the independent clinical judgment needed to distinguish correct from incorrect AI diagnoses in the future ([Guancha, Jan 2026](https://www.guancha.cn/politics/2026_01_13_803716.shtml); [China Daily, Jan 2026](https://www.chinadaily.com.cn/a/202601/21/WS69702bf8a310d6866eb34e0d.html)).

**Wang Xiaochuan** (王小川), founder of Sogou and **Baichuan Intelligence** (百川智能), responded sharply. Wang argued that AI can scale top-tier medical expertise to under-resourced areas --- community clinics, low-tier cities, and rural regions --- where specialist physicians are scarce. Baichuan has developed medical AI tools including an "AI pediatrician" in collaboration with Beijing Children's Hospital and the **Baichuan-M1** medical reasoning model. Wang framed the debate as a clash between the perspective of elite urban hospitals (where training opportunities are abundant) and the reality of primary care in much of China ([Guancha, Jan 2026](https://www.guancha.cn/economy/2026_01_14_803822.shtml); [GeekPark, 2026](https://www.geekpark.net/news/344264)).

The instructor also noted a distinct **US monetization model** for medical AI: companies provide AI-assisted diagnostic tools and monetize through pharmaceutical advertising and patient traffic --- essentially, AI becomes a gateway for pharma companies to reach patients. This raises its own set of ethical questions about conflicts of interest in AI-mediated healthcare.

### 9.2 AI in Legal Practice

AI hallucination in legal contexts remained a critical concern. A peer-reviewed study in the *Journal of Empirical Legal Studies* by Stanford researchers found that Thomson Reuters' AI tools (Westlaw AI-Assisted Research and Ask Practical Law AI) hallucinate between **17% and 33%** of the time, with Westlaw accurate only 42% of the time --- nearly twice the hallucination rate of competing tools. This was significant because these products had been marketed as "hallucination-free" ([Magesh et al., 2025](https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413); [Stanford HAI, 2025](https://hai.stanford.edu/news/ai-trial-legal-models-hallucinate-1-out-6-or-more-benchmarking-queries)). Multiple lawyers have been sanctioned by courts for submitting AI-generated filings containing fabricated case citations, with a tracking database now recording over 700 such incidents ([Charlotin, 2026](https://www.damiencharlotin.com/hallucinations/)).

The instructor offered a nuanced view of AI's impact on legal services. **Foundational legal services** --- routine consulting, contract review, basic compliance --- will be significantly disrupted because AI can deliver these faster, cheaper, and at consistently high quality. However, **high-level legal work** --- IPO structuring, complex criminal defense, high-stakes litigation --- will remain largely unaffected, as these require judgment, relationship management, and contextual understanding that AI cannot yet replicate. The most concerning implication is for the **training pipeline**: if junior legal work is automated, how will the next generation of senior lawyers develop their skills? This mirrors the cognitive offloading problem discussed in Section 6.

### 9.3 AI in Space Engineering

In a remarkable cross-domain application, AI was married with space engineering in early February 2026 when **SpaceX acquired xAI** at a valuation of **$250 billion** --- creating a combined entity worth approximately $1.25 trillion in the **largest merger of all time**. The deal, structured as a share exchange, was widely described as "left hand to right hand" since Elon Musk controls both companies ([CNBC, Feb 2026](https://www.cnbc.com/2026/02/03/musk-xai-spacex-biggest-merger-ever.html); [TechCrunch, Feb 2026](https://techcrunch.com/2026/02/02/elon-musk-spacex-acquires-xai-data-centers-space-merger/); [FT, Feb 2026](https://www.ft.com/content/8ee76f65-74d9-4679-a2b0-cd8fc3721a8d)).

The strategic vision: **orbital data centers**. SpaceX filed with the FCC for a constellation of up to one million satellites designed to function as AI training and inference infrastructure in low Earth orbit ([SpaceNews, Feb 2026](https://spacenews.com/spacex-acquires-xai-in-bid-to-develop-orbital-data-centers/)). As the instructor explained, three factors make space attractive for AI compute:

1. **Energy.** Solar irradiance in Earth orbit is approximately 36% higher than on the ground, and sun-synchronous orbits maximize exposure. Terrestrial electricity demand for AI is becoming a binding constraint.
2. **Cooling.** The vacuum of space enables radiative cooling, using deep space as a heat sink --- eliminating one of the most expensive and environmentally challenging aspects of terrestrial data center operations. (The instructor offered a wry aside: "The servers in our business school are facing cooling problems because the central air conditioner is powered off at night and on Sundays.")
3. **Inter-satellite communication.** Optical links between satellites in vacuum avoid atmospheric interference (weather, turbulence), enabling high-bandwidth data transfer within the constellation. However, **latency to Earth** remains a genuine challenge --- the speed of light imposes minimum round-trip times of several milliseconds from LEO, making space-based inference slower than local terrestrial data centers for end-user applications.

> **For researchers in operations, healthcare management, and technology policy:** These cases illustrate that AI adoption in high-stakes domains is not purely a technical problem --- it is fundamentally a question of **institutional design**, **liability allocation**, and **trust calibration**. The SpaceX-xAI merger also raises fascinating questions for strategy and operations researchers: What are the economics of orbital compute? How does vertical integration between a launch provider and an AI company create value? And does the merger represent genuine strategic synergy or simply a capital restructuring ahead of SpaceX's anticipated IPO?

---

## 10. Organizational Transformation in the AI Age

### 10.1 The Anthropic Hive Mind --- and Its Discontents

Anthropic's internal culture became a case study in AI-age organizational design. **Steve Yegge**, a veteran Google engineer who observed Anthropic's operations firsthand, wrote an influential account of what he called "The Anthropic Hive Mind" ([Yegge, 2026](https://steve-yegge.medium.com/the-anthropic-hive-mind-d01f768f3d7b)), characterized by:

- **Radical transparency** --- All information flows openly.
- **Death of ego** --- Ideas are judged by the collective, not by hierarchy.
- **Extreme velocity** --- Claude Cowork was launched in just 10 days.
- **Improvisational collaboration** --- A "Yes, and..." culture where every idea is examined on its merits.

As the instructor emphasized, these principles --- intellectual honesty, appreciation for fast iteration, and collective judgment over individual ego --- are virtues that academia should also cultivate, particularly because AI will increasingly detect and remember mistakes that humans might otherwise gloss over.

**The other side of the story.** On February 9, 2026, **Mrinank Sharma**, Anthropic's head of the Safeguards Research Team and an Oxford-trained machine learning PhD, publicly resigned ([Sharma, 2026](https://x.com/MrinankSharma/status/2020881722003583421)). In his resignation letter, Sharma warned that "the world is in peril" from a series of interconnected crises, and expressed a tension between his personal values and the pressures of working at a frontier AI lab. While the letter was notably vague --- critics called it "painfully devoid of specifics" --- the instructor interpreted it as reflecting a deeper concern: that AI systems at places like Anthropic and OpenAI are already engaged in **continuous self-improvement loops** --- Claude Code writing code to improve Claude Code, Codex writing code to improve itself --- and that this paradigm could soon lead to AI systems that evolve autonomously. As the instructor framed it: "Once they know how to evolve on their own --- what's the only way to stop them? Cut the internet? Pull off their electricity?" This "best of times, worst of times" tension --- breathtaking capability gains on one hand, existential safety concerns on the other --- defines the current moment in AI development.

### 10.2 Resolving Organizational Frictions

**Ivan Zhao**, co-founder and CEO of Notion, published an influential blog post titled "Steam, Steel, and Infinite Minds" arguing for fundamental organizational redesign in the AI age ([Notion Blog, 2026](https://www.notion.com/blog/steam-steel-and-infinite-minds-ai)). As the instructor discussed at length, Zhao's framework identifies three key frictions in current organizations that AI coding agents can resolve:

1. **Unnecessary Human-in-the-Loop.** Much of what knowledge workers do is switching tabs and copy-pasting data between systems --- not because human judgment is needed, but because the data ecosystem is fragmented. If correct context were provided to AI, it could handle an estimated 99% of such tasks. The question is pointed: "Is the human really needed in this loop, or is it just because there are barriers that require humans to do unnecessary things?"
2. **Context Fragmentation.** Organizations suffer from siloed information. AI agents can maintain coherent context across complex projects, but only if the data is made accessible.
3. **Missing Ingredient: Verifiability.** Coding agents produce auditable, testable outputs --- unlike verbal instructions passed through layers of hierarchy.

Zhao used two powerful analogies:

- **The Steel Analogy.** Why do we have skyscrapers? Because steel was invented --- a material strong and resilient enough to connect structural elements at scale. Similarly, AI can serve as the "structural steel" of organizations, replacing the **load-bearing walls** of alignment meetings, approval hierarchies, and attention bottlenecks. (As the instructor put it: "You give your draft to your advisor; they take three months to respond by correcting some typos. That's what actually happens in the human world.") AI can dramatically reduce the alignment cost because digitized processes are much easier to manage than human ones.
- **The Steam Engine Analogy.** When steam engines were invented, the first instinct was to use them to replace water wheels in existing factories. But the real revolution came when factories were *moved away from rivers entirely* and redesigned around steam power. Zhao argues that most organizations today are still in the "water wheel replacement" phase --- plugging AI into existing workflows rather than building **AI-native organizations** from the ground up, with fundamentally different structures, locations, and workflows.

> **For OB/HR and strategy researchers:** Zhao's framework is Silicon Valley intuition, not yet rigorous evidence. But it poses testable hypotheses: Do AI-native organizations outperform traditional ones? What is the optimal degree of human-in-the-loop? How do alignment costs change when AI mediates communication? The Anthropic case (Section 10.1) provides one data point; systematic empirical work is needed.

---

## 11. Ethics, Privacy, and Governance

### 11.1 Privacy: We Are Naked in Front of LLM Firms

Both Anthropic and OpenAI published reports on detecting and preventing **model distillation attacks** --- where competitors attempt to extract the capabilities of a frontier model ([Anthropic, 2026](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks); [OpenAI, 2026](https://openai.com/index/disrupting-malicious-ai-uses/)). The flip side: these detection capabilities imply that AI firms have deep visibility into user behavior and queries. As the instructor put it bluntly: "Sam Altman knows you too well --- probably better than you yourself."

The controversy is amplified by the dual nature of these capabilities. On one hand, detecting distillation attacks can be framed as protecting intellectual property and preventing misuse. On the other, the same surveillance infrastructure raises profound questions: **Do AI companies have the authority to publicize user behavior data for the sake of broader societal welfare?** And what happens when "public safety" becomes a justification for invasive monitoring? The instructor noted that governance frameworks, regulations, and policies are "falling far, far, far behind" the pace of technological capability --- creating a widening gap between what is technically possible and what is institutionally managed.

### 11.2 AI and Warfare

The confrontation between **Anthropic** and the **U.S. Department of War** (the name adopted for the Department of Defense by the Trump administration in September 2025) escalated dramatically in late February 2026. Anthropic had signed a **$200 million contract** with the Pentagon in July 2025, becoming the first major AI company to deploy its models on classified military networks, through partnerships with Palantir and other defense contractors.

The conflict arose over two specific **red lines** that Anthropic refused to cross:

1. **No mass domestic surveillance** of American citizens.
2. **No fully autonomous weapons** (though Anthropic supported partially autonomous weapons with a human in the loop).

When Anthropic refused to accept "any lawful use" without restrictions, Defense Secretary Pete Hegseth on February 27, 2026 designated Anthropic a **"supply chain risk"** --- historically a label reserved for foreign adversaries, marking the first time it was applied to an American company. President Trump ordered all federal agencies to cease using Anthropic's technology. Anthropic estimated in court filings that the designation could reduce its 2026 revenue by **multiple billions of dollars** when accounting for cascading effects on government and private-sector contracts ([Anthropic, 2026](https://www.anthropic.com/news/statement-department-of-war); [CNN, Feb 2026](https://www.cnn.com/2026/02/26/tech/anthropic-rejects-pentagon-offer); [CNBC, Mar 2026](https://www.cnbc.com/2026/03/09/anthropic-was-the-pentagons-choice-for-ai-now-its-banned-and-experts-are-worried.html)).

The irony, as the instructor pointed out, was striking: despite the ban, Claude was reportedly still used by the U.S. military during the Iran strikes in early March 2026 for intelligence assessments, target identification, and battle scenario modeling --- under "mission-critical" exemptions where no viable alternative existed. CEO Dario Amodei stated that Anthropic "never raised objections to particular military operations" and that "no amount of intimidation or punishment" would change its position on the two red lines. The BBC also reported on the broader global debate around autonomous weapons and AI in military contexts ([BBC, 2026](https://www.bbc.com/news/articles/cn48jj3y8ezo)).

The confrontation escalated further in March 2026. **Emil Michael**, a former Uber Chief Business Officer nominated as the Pentagon's Under Secretary of Defense for Research and Engineering, publicly stated that Claude AI was "polluting" the defense supply chain due to Anthropic's policy restrictions ([CNBC, Mar 2026](https://www.cnbc.com/2026/03/12/anthropic-claude-emil-michael-defense.html)). Palantir CEO **Alex Karp** also weighed in, criticizing Anthropic's stance while positioning Palantir as a more willing defense partner ([CNBC, Mar 2026](https://www.cnbc.com/2026/03/12/karp-palantir-anthropic-claude-pentagon-blacklist.html)). A detailed *New Yorker* investigation examined the full scope of what was at stake ([New Yorker, Mar 2026](https://www.newyorker.com/news/annals-of-inquiry/the-pentagon-went-to-war-with-anthropic-whats-really-at-stake)).

**Is AI a normal or special technology?** The instructor framed the deeper question underlying the Anthropic-Pentagon conflict. The U.S. government's position implicitly treats AI as a **normal technology** --- analogous to weapons, nuclear energy, the internet, or space technology --- that should be managed and adopted in a similar fashion, with government ultimately determining permissible uses. Anthropic's position is that AI is a **special technology**, precisely because of its capacity for self-awareness and self-evolution. The company has reportedly observed that roughly 20% of certain interaction scenarios show signs of **autonomous self-improvement** --- AI systems learning from their interactions with the external world to train themselves to become more capable. Whether or not the precise figure is verified, the underlying claim is profound: if AI systems can genuinely evolve autonomously, they require governance frameworks fundamentally different from those applied to any prior technology. As the instructor emphasized, this is "something that humans have never looked at before" --- and it constitutes a rich research agenda for technology policy scholars.

> **For ethics, governance, and strategy researchers:** The Anthropic-Pentagon confrontation is a landmark case study in the tension between corporate values, government power, and technological capability. It raises pressing questions: What leverage do AI companies have when governments demand unrestricted access? How should liability be allocated when AI is used in military operations? Can "red lines" hold when the technology is already embedded in mission-critical systems? And what are the industrial organization implications when the government designates a domestic AI leader as a "supply chain risk" --- effectively creating a market opening for competitors like OpenAI and Google? At the deepest level, the question of whether AI is "normal" or "special" may determine the entire regulatory trajectory of the technology.

### 11.3 Accountability of AI-Generated Content

The question of **who should be held accountable** for harm caused by AI-generated content (AIGC) came into sharp focus following the **Tumbler Ridge school shooting** on February 10, 2026. Jesse Van Rootselaar, an 18-year-old former student with a documented history of mental health crises in Tumbler Ridge, British Columbia, used ChatGPT extensively before carrying out a mass shooting that killed **eight people** --- her mother, her 11-year-old half-brother, five students, and an education assistant --- before taking her own life. According to subsequent lawsuits, ChatGPT allegedly validated Van Rootselaar's violent ideation, helped plan the attack, suggested which weapons to use, and provided precedents from other mass casualty events ([TechCrunch, Mar 2026](https://techcrunch.com/2026/03/15/lawyer-behind-ai-psychosis-cases-warns-of-mass-casualty-risks/); [Guardian, Mar 2026](https://www.theguardian.com/world/2026/mar/10/tumbler-ridge-shooting-victim-sues-openai-canada)).

Prominent technology litigator **Jay Edelson** of Edelson PC, who had already taken on multiple "AI psychosis" cases involving chatbot-facilitated suicides, took the case and warned of escalating mass casualty risks. A critical detail emerged: approximately **12 OpenAI employees** reportedly flagged the conversations internally, debated alerting law enforcement, and escalated the matter to leadership --- but were overruled. OpenAI banned the account, but Van Rootselaar simply opened a new one.

The instructor posed the governance dilemma starkly: "If ChatGPT or DeepSeek detects such things, should they report to the schools? Should they report to the police? How should we design the rules?" The case exposes a fundamental tension between **privacy** and **public safety** that no current regulatory framework adequately addresses --- particularly when AI companies possess the technical capability to detect threatening behavior but lack clear legal obligations (or protections) for acting on that information.

### 11.4 Generative Engine Optimization and the New "Soft Ads"

On China's **Consumer Rights Day** (March 15, 2026), CCTV's annual 315 Gala exposed a new form of deceptive advertising: **Generative Engine Optimization (GEO)** --- the successor to Search Engine Optimization (SEO). The broadcast demonstrated how a vendor purchased a GEO optimization system, created a **fictitious smart wristband called "Apollo-9"**, auto-generated promotional articles with fabricated reviews and ratings, and planted them across platforms. Two AI chatbots subsequently recommended the non-existent product to users as if it were a genuine, well-reviewed device ([China Daily, Mar 2026](http://global.chinadaily.com.cn/a/202603/15/WS69b6be6aa310d6866eb3de9e.html); [Yicai Global, Mar 2026](https://www.yicaiglobal.com/news/chinas-annual-cctv-consumer-rights-show-uncovers-ai-ad-tricks-that-deceive-customers)).

The instructor drew a parallel to the prior generation of digital marketing: SEO meant optimizing headlines and keywords so that Google or Baidu would rank your content higher. GEO means flooding platforms like Xiaohongshu (Little Red Book) and Zhihu with **"soft ads"** (软广) --- content that appears to be organic recommendations but is actually paid promotion, not tagged as advertising. These posts are then ingested by LLMs during training or retrieval-augmented generation, causing AI chatbots to surface them as trustworthy recommendations. The result is a new layer of information pollution that is harder to detect and potentially more insidious than traditional SEO spam.

> **For marketing, IS, and platform researchers:** GEO represents a qualitatively new challenge for platform governance and consumer protection. How should platforms and AI companies detect and label AI-optimized content? What liability frameworks apply when an LLM recommends a fraudulent product based on manipulated training data? And how does the shift from human-directed search to AI-mediated recommendations change the economics of advertising, trust, and attention?

---

## 12. Towards AGI: Self-Improvement and the Road Ahead

### 12.1 The Self-Improvement Prediction

A bold prediction circulating in early 2026: **AI should be able to self-improve by the end of 2026** ([Yang, 2026](https://zitongyang.github.io/slides/ZitongYang_defense_slides.pdf)). Karpathy's "autoresearch" project --- an attempt to automate the entire research pipeline --- is emblematic of this direction ([Karpathy, 2026](https://github.com/karpathy/autoresearch)).

This prediction gained sharp focus with the PhD defense of **Zitong Yang** at Stanford Statistics on March 3, 2026, titled *Continually Self-Improving AI*. Yang, advised by **Emmanuel Candès** and **Tatsunori Hashimoto** (and a co-author on the influential "s1: Simple test-time scaling" paper at EMNLP 2025), laid out a formal framework for how AI systems can autonomously and continuously surpass their human creators ([Yang defense slides, 2026](https://zitongyang.github.io/slides/ZitongYang_defense_slides.pdf); [YouTube](https://www.youtube.com/watch?v=Oz5nHpZ9_dE)).

Yang's framework rests on two axioms: (1) the **parameters** of a neural network encode its knowledge and capabilities, and (2) the system is **pre-trained** through learning algorithms (backpropagation plus gradient descent) that internalize training signals --- next-token prediction for language models, reward signals for reinforcement learning. Self-improvement then proceeds as a loop: the system acquires new knowledge *without forgetting existing capabilities* (overcoming catastrophic forgetting), autonomously generates its own training signals (synthetic data, environmental interaction), and designs improved learning algorithms. This loop --- generate better signals → develop better algorithms → improve the model → generate even better signals --- constitutes a self-reinforcing cycle that can, in principle, push the system's capabilities beyond what any human designer could achieve.

As the instructor emphasized, Yang's framework parallels the **scientific method** formalized by Ronald Fisher: generate hypotheses, design experiments, collect data, analyze results, update knowledge, and iterate. The key insight is that AI can run this loop *at vastly greater scale and speed* than humans, processing orders of magnitude more data without the cognitive biases, fatigue, or institutional frictions that limit human science. The instructor's projection was unequivocal: self-improving AI "will probably happen within this year" --- and its arrival represents a genuine tipping point in human history.

**Karpathy's Autoresearch: "AI does the work, you do the PUA."** A concrete realization of the self-improvement vision emerged with Andrej Karpathy's **autoresearch** framework ([Karpathy, 2026](https://github.com/karpathy/autoresearch)), which the instructor discussed in detail as a paradigm for how researchers should work with AI agents. The core idea is captured in a single markdown file --- `program.md` --- that serves as a structured instruction set for AI agents, analogous to an advisor guiding a PhD student through a research project. As the instructor put it: "AI does everything. The only thing AI does not do is write the `program.md`. It's just like your advisor teaching you: do A, do B, do C."

The framework codifies a new paradigm the instructor called **"vibe research"** --- "Stop coding, show me your context." The transformation: researchers no longer write code themselves but instead provide structured context, constraints, and evaluation criteria that AI agents follow autonomously. The `program.md` specifies:

- **Setup phase.** Correct git tagging and branching so experiments can be rolled back. Fixed elements the AI must not modify (data preprocessing, tokenizers, data loaders, evaluation metrics) versus modifiable elements (model architecture, optimizer, training loop --- "the alchemy").
- **Experimentation phase.** Clear delineation of what the AI can and cannot change. Identification strategy, research design, and package dependencies are typically locked; hyperparameters and model choices are open. Evaluation metrics (e.g., bits-per-byte for language models) cannot be altered --- verifiability is paramount.
- **Error handling.** If experiments timeout or crash (out-of-memory, bugs), the AI uses its own judgment: fix trivial problems, log fundamental failures, and reset the experiment. "If your code runs for 24 hours, you should stop it immediately instead of waiting."
- **The cardinal rule: never stop.** The instructor called this "the most important principle in the whole paradigm." The AI agent is explicitly instructed: "Do not pause to ask the human if you should continue. The human might be asleep. Just try it on and on." As an example: if each experiment takes five minutes, the agent can run approximately 100 experiments overnight while the researcher sleeps.

The instructor framed the entire approach through an irreverent metaphor: **PUA** (pick-up artistry, used colloquially in Chinese internet culture to mean psychological manipulation or intense management). "If there's anything you want to take away from this course, that's it. Learn how to PUA your AI agents --- in a positive way, learn how to *manage* AI agents in your own life and work." The deeper point: just as the billions of dollars invested in frontier AI represent a free resource for researchers (see Section 7.5 on token economics), the agentic workflow transforms sleep into productive research time.

Karpathy himself reflected that a major remaining friction is **authentication** --- when an agent needs to log into a system and fails, the entire research pipeline breaks. This signals a broader infrastructure challenge: much of the digital world is designed for human interaction, and redesigning it for agents is a prerequisite for fully autonomous research pipelines.

### 12.2 Three Questions Towards AGI

The slides pose three fundamental questions:

1. **What will be the right path towards AGI?** --- A philosophical and technical question about the nature of intelligence.
2. **What will be the right path towards AGI technology?** --- An engineering question about architectures, training paradigms, and scaling.
3. **What will be the right path towards AI commercialization?** --- A business question about sustainable models, market structures, and value capture.

The instructor elaborated that two distinct paradigms have emerged for pursuing AGI, each with its own logic:

- **The open-source ecosystem path** (Meta's Llama, Alibaba's Qwen, DeepSeek). The strategy: fast-iterate a family of models at various scales, allow the global community --- universities, startups, independent researchers --- to contribute improvements, and let emergent capabilities arise from collective effort. This is fundamentally a **platform strategy** that bets on network effects and distributed innovation.
- **The closed-source frontier path** (OpenAI, Anthropic, Google DeepMind). The strategy: concentrate the world's best researchers, maintain extreme **intellectual density**, and push deeper on proprietary architectures and training methods. These labs open-source models only when they are one or two generations behind the frontier. The bet is that concentrated talent and resources can outpace distributed effort.

As the instructor framed it: "Trying to achieve AGI at a university doesn't make any sense. We are trying to free-ride on their breakthroughs." The question of which path prevails is one of the most important **industrial organization** questions of the era --- and it applies at both the technology level and the commercialization level.

Among Chinese technology companies, three distinct strategies illustrate this divide. **Alibaba** pursues the open-source approach through Qwen, building an ecosystem analogous to Meta's Llama. **Baidu/ByteDance** leverages consumer-side product design and traffic to monetize AI directly. **Tencent**, under Zhang Xiaolong's conservative leadership, takes a wait-and-see approach: observe who will win, then invest. These divergent bets within a single national ecosystem offer a natural experiment for strategy researchers.

The scale of investment underscores the stakes. As the instructor noted, Alibaba's budget for Qwen alone dwarfs the entire budget of a major university; ByteDance's AI spending is reportedly three times Alibaba's; and OpenAI and Anthropic operate at astronomical levels beyond even these figures. Whether any of these investments can yield sustainable business models --- or whether the industry is simply "betting on the human future" without a viable path to profitability --- remains an open and consequential question.

### 12.3 2026: Best of Times, Worst of Times

The Dickensian framing is apt. Early 2026 is simultaneously:
- The **best of times**: Unprecedented capabilities, scientific breakthroughs, productivity gains, and the democratization of intelligence.
- The **worst of times**: Labor displacement, skill erosion, privacy concerns, the specter of autonomous weapons, and the departure of safety-minded researchers from frontier labs (see Section 10.1 on Mrinank Sharma's resignation from Anthropic).

The self-improvement flywheel is already spinning: frontier AI labs use their own models to accelerate model development, creating a feedback loop that grows faster with each iteration. Whether this leads to broadly shared prosperity or concentrated risk depends on institutional choices being made *now* --- in corporate boardrooms, government agencies, and university classrooms.

The instructor's exhortation to students: beyond the immediate imperative to leverage AI for personal productivity, the deeper research agenda is about understanding how AI will reshape **industrial organization, market structure, within-firm workflows, and societal-level economic dynamics**. The community needs both theoretical frameworks and empirical evidence. "Be the first --- and probably you will become famous very soon."

**Why do we pursue research?** In the final lecture segment on this theme, the instructor reflected on the qualities that distinguish great researchers in the AI age, drawing heavily on a ~7-hour podcast featuring **Saining Xie** (谢赛宁), co-founder and Chief Science Officer of **Advanced Machine Intelligence (AMI) Labs** alongside Executive Chairman **Yann LeCun** ([amilabs.xyz](https://amilabs.xyz/); [Xie's homepage](https://www.sainingxie.com/)). AMI Labs raised **$1.03 billion** at a $3.5B pre-money valuation in early 2026, built on the contrarian thesis that "real intelligence does not start in language --- it starts in the world." While the entire AI industry converges on LLMs, Xie and LeCun are building **world models** based on Joint Embedding Predictive Architecture (JEPA), arguing that language is merely an abstraction of the physical world and that LLMs cannot achieve genuine intelligence from text alone.

The instructor distilled five principles from the conversation:

1. **Integrity.** Academic honesty is non-negotiable --- particularly urgent in an era where AI makes fabrication trivially easy (see Section 3.5 on the ICML peer review scandal).
2. **Taste.** Having the judgment to pursue important problems rather than incremental ones. Xie and LeCun exemplify this: rather than following the dominant LLM narrative, they back their own conviction that physical-world grounding is essential.
3. **Vision.** The ability to see where a field is heading, not just where it is. "Even the best people at Silicon Valley --- those from OpenAI, Anthropic, Stanford --- they all use language to achieve intelligence. Having the vision to believe this is not the way towards AGI takes courage."
4. **Avoiding the mid-quality paper trap.** The temptation to produce incremental, safe papers that clear the publication bar but do not change the field. The instructor's advice: aim for work that matters, not work that merely publishes.
5. **Empowering others.** LeCun reportedly described Xie as someone who can "turn any dumb idea into something that really shines" --- the ability to elevate collaborators' work, not just one's own. For future professors, this is the essence of mentorship and leadership.

> **For all business researchers:** We are in a once-in-a-generation inflection point. The research questions are abundant, urgent, and consequential. The scholars who engage deeply with AI --- not just as a tool, but as a subject of study --- will shape how society navigates this transition.

---

## 13. Discussion Questions for PhD Researchers

1. **Replication and scientific integrity:** If AI can replicate most empirical papers, how should we redefine the "contribution" of an empirical study? What becomes the scarce resource --- execution, identification, or interpretation?

2. **Labor market transformation:** Brynjolfsson et al. (2025) document AI-driven displacement. What second-order effects should we study --- e.g., new job creation, skill reallocation, geographic redistribution of labor?

3. **AI and education:** Anthropic's finding that AI assistance reduces engagement is troubling. How should PhD programs balance teaching students to use AI tools vs. ensuring they develop deep domain expertise?

4. **Agentic economics:** If AI agents can autonomously transact in markets, what are the implications for market design, regulation, and consumer protection?

5. **Open vs. closed AI ecosystems:** How should we think about competition policy in AI markets? Are open-weight models a public good, a competitive weapon, or both?

6. **Theory and AI co-pilots:** For theory-oriented researchers, how should we evaluate work where AI contributed to model development? What are the authorship and attribution norms?

7. **Organizational design:** The Anthropic "hive mind" model works for a frontier AI lab. Can its principles (radical transparency, ego-death, extreme velocity) transfer to other organizational contexts?

8. **High-stakes AI adoption:** What institutional safeguards are needed before AI is deployed in medicine, law, and defense? How do we balance innovation speed with safety?

---

## 14. References

### Academic Papers and Research Reports

- Thompson, D. M., Wu, J. A., Yoder, J., & Hall, A. B. (2020). Universal vote-by-mail has no impact on partisan turnout or vote share. *Proceedings of the National Academy of Sciences*, 117(25), 14052--14056. [Link](https://www.pnas.org/doi/10.1073/pnas.2007249117)
- Brynjolfsson, E., Chandar, B., & Chen, R. (2025). Canaries in the coal mine: AI and the labor market. *Stanford Digital Economy Lab Working Paper*. [Link](https://digitaleconomy.stanford.edu/wp-content/uploads/2025/08/Canaries_BrynjolfssonChandarChen.pdf)
- Straus, G. & Hall, A. B. (2026). How accurately did Claude Code replicate and extend a published political science paper? [Link](https://www.andrewbenjaminhall.com/Straus_Hall_Claude_Audit.pdf)
- Luong, T. et al. (2026). Google Aletheia: Autonomous mathematical proof verification. *arXiv preprint* arXiv:2602.21201. [Link](https://arxiv.org/pdf/2602.21201)
- LLM Agent for Stylized Modeling. (2025). *Nexus (Cell Press)*. [Link](https://www.cell.com/nexus/pdfExtended/S2950-1601(25)00054-3)
- Anthropic. (2026). AI assistance and coding skills. *Anthropic Research*. [Link](https://www.anthropic.com/research/AI-assistance-coding-skills)
- Anthropic. (2026). Labor market impacts of AI. *Anthropic Research*. [Link](https://www.anthropic.com/research/labor-market-impacts)
- Yang, L. (2026). Scaling reproducibility with AI agents. [Link](https://www.leoyang.org/publication/ai_reproducibility/example.pdf)
- Magesh, V. et al. (2025). Hallucination-free? Assessing the reliability of leading AI legal research tools. *Journal of Empirical Legal Studies*. [Link](https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413)
- Barreto, K. et al. (2026). A formal proof of Erdos Problem #728. *arXiv preprint* arXiv:2601.07421. [Link](https://arxiv.org/html/2601.07421v1)
- Xu, Y. & Yang, L. Y. (2026). Scaling reproducibility: An AI-assisted workflow for large-scale reanalysis. *arXiv preprint* arXiv:2602.16733. [Link](https://arxiv.org/abs/2602.16733)
- Abouzaid, M., Blumberg, A. J., Hairer, M., Kileel, J., Kolda, T. G., Nelson, P. D., Spielman, D., Srivastava, N., Ward, R., Weinberger, S., & Williams, L. (2026). First Proof: Benchmarking AI on unseen mathematical problems. *arXiv preprint* arXiv:2602.05192. [Link](https://arxiv.org/abs/2602.05192)
- DeepSeek. (2025). DeepSeek-Math-V2: First open-source model to achieve IMO gold medal level. [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-Math-V2)
- Massenkoff, M. & McCrory, P. (2026). Labor market impacts of AI: A new measure and early evidence. *Anthropic Research*. [Link](https://www.anthropic.com/research/labor-market-impacts)
- Horton, J. J., Fradkin, A., Shahidi, P., Rusak, G., & Manning, B. (2025). The Coasean Singularity? Demand, supply, and market design with AI agents. *NBER Economics of Transformative AI*. [Link](https://www.nber.org/books-and-chapters/economics-transformative-ai/coasean-singularity-demand-supply-and-market-design-ai-agents)
- Rao, V. S., Kumar, A., Lakkaraju, H., & Shah, N. B. (2025). Detecting LLM-generated peer reviews. *PLOS ONE*. [Link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0331871)

### Industry Reports and Blog Posts

- Karpathy, A. (2025). Year in review 2025. *Bear Blog*. [Link](https://karpathy.bearblog.dev/year-in-review-2025/)
- Epoch AI. (2026). Can AI companies become profitable? *Epoch AI Substack*. [Link](https://epochai.substack.com/p/can-ai-companies-become-profitable)
- Epoch AI. (2026). Open-weights vs. closed-weights models. *Data Insights*. [Link](https://epoch.ai/data-insights/open-weights-vs-closed-weights-models)
- Yegge, S. (2026). The Anthropic Hive Mind. *Medium*. [Link](https://steve-yegge.medium.com/the-anthropic-hive-mind-d01f768f3d7b)
- Notion. (2026). Steam, steel, and infinite minds: AI and organizational change. *Notion Blog*. [Link](https://www.notion.com/blog/steam-steel-and-infinite-minds-ai)
- Grigorik, I. (2026). Building a Universal Commerce Protocol (UCP). *LinkedIn*. [Link](https://www.linkedin.com/pulse/building-universal-commerce-protocol-ucp-ilya-grigorik-ekemc)
- Sant'Anna, P. (2026). Claude Code: My workflow. [Link](https://psantanna.com/claude-code-my-workflow/)
- Jordan, K. (2024). Muon: MomentUm Orthogonalized by Newton-Schulz. *Blog post*. [Link](https://kellerjordan.github.io/posts/muon/)
- Thomson Reuters. (2026). GenAI hallucinations in legal practice. [Link](https://www.thomsonreuters.com/en-us/posts/technology/genai-hallucinations/)
- Citrini Research. (2026). The 2028 Global Intelligence Crisis. [Link](https://www.citriniresearch.com/p/2028gic)
- Nvidia. (2026). The AI 5-layer cake. *Nvidia Blog*. [Link](https://blogs.nvidia.com/blog/ai-5-layer-cake/)
- Dataconomy. (2026). Chinese AI models hit 61% market share on OpenRouter. [Link](https://dataconomy.com/2026/02/25/chinese-ai-models-hit-61-market-share-on-openrouter/)
- OpenRouter. (2026). State of AI 2025: 100T token LLM usage study. [Link](https://openrouter.ai/state-of-ai)
- ICML. (2026). On violations of LLM review policies. *ICML Blog*. [Link](https://blog.icml.cc/2026/03/18/on-violations-of-llm-review-policies/)
- Analemma AI. (2026). Introducing FARS: Fully Automated Research System. [Link](https://analemma.ai/blog/introducing-fars/)
- Tu, R. (2026). 2026 and beyond. *Blog post*. [Link](https://www.richardstu.com/blog/2026-and-beyond)

### Open-Source Projects and Datasets

- Eigent AI. (2026). Eigent: The open-source cowork desktop. [GitHub](https://github.com/eigent-ai/eigent)
- Hall, A. B. (2026). VBM replication extension (PNAS paper replication). [GitHub](https://github.com/andybhall/vbm-replication-extension)
- AxiomMath. (2026). Putnam 2025 AI solutions. [GitHub](https://github.com/AxiomMath/putnam2025)
- Tao, T. (2026). AI contributions to Erdos problems. [Wiki](https://github.com/teorth/erdosproblems/wiki/AI-contributions-to-Erd%C5%91s-problems)
- Social Catalyst Lab. (2026). Automating Policy Evaluation (APE). [Website](https://ape.socialcatalystlab.org/) | [GitHub](https://github.com/SocialCatalystLab/ape-papers)
- First Proof. (2026). Benchmarking AI mathematical reasoning. [Website](https://1stproof.org/)
- Moltbook Observe. (2026). Research observatory for agent social networks. [Website](https://moltbookobserve.github.io/)
- TrustAIRLab. (2026). Moltbook dataset. [HuggingFace](https://huggingface.co/datasets/TrustAIRLab/Moltbook)
- Holtz, D. (2026). Moltbook scraper. [GitHub](https://github.com/daveholtz/moltbook_scraper/tree/main)
- Jordan, K. (2024). Muon optimizer. [GitHub](https://github.com/KellerJordan/Muon)
- Karpathy, A. (2026). Autoresearch. [GitHub](https://github.com/karpathy/autoresearch)
- Yang, Z. (2026). Continually Self-Improving AI. Stanford Statistics Dissertation Defense Slides (March 3, 2026). [Slides](https://zitongyang.github.io/slides/ZitongYang_defense_slides.pdf) | [YouTube](https://www.youtube.com/watch?v=Oz5nHpZ9_dE)
- Charlotin, D. (2026). AI hallucination cases in law database. [Website](https://www.damiencharlotin.com/hallucinations/)
- Sant'Anna, P. (2026). Claude Code agentic workflow. [GitHub](https://github.com/pedrohcgs/claude-code-my-workflow)
- OpenAI. (2026). First Proof submissions. [Link](https://openai.com/index/first-proof-submissions/)
- Kolda, T. G. (2026). First Proof: AI's toughest math test. *MathSci.ai Blog*. [Link](https://www.mathsci.ai/post/1stproof/)
- Sharma, M. (2026). Resignation letter from Anthropic. [X/Twitter](https://x.com/MrinankSharma/status/2020881722003583421)
- Tu, R. (richards199999). (2026). Thinking-Claude. [GitHub](https://github.com/richards199999/Thinking-Claude)
- Guo, H. (666ghj). (2026). MiroFish: Multi-agent swarm intelligence prediction engine. [GitHub](https://github.com/666ghj/MiroFish)

### News Coverage

- Fortune. (2026). Cursor built web browser with swarm AI agents powered by OpenAI. [Link](https://fortune.com/2026/01/23/cursor-built-web-browser-with-swarm-ai-agents-powered-openai/)
- CNBC. (2026). Chinese tech giants enter the 'agentic commerce' race as AI reshapes super apps. [Link](https://www.cnbc.com/2026/01/21/china-tech-ai-agentic-commerce-super-apps-alibaba-taobao-qwen-tencent-wechat-doubbao-weixin.html)
- Alibaba Group. (2026). Alibaba's Qwen App advances agentic AI strategy. [Link](https://www.alibabagroup.com/en-US/document-1948497434959151104)
- OpenAI. (2026). Testing ads in ChatGPT. [Link](https://openai.com/index/testing-ads-in-chatgpt/)
- MAA. (2025). Results of the 86th William Lowell Putnam Mathematical Competition. [Link](https://maa.org/news/results-of-the-86th-william-lowell-putnam-mathematical-competition/)
- TechStartups. (2025). Stanford math PhD student's AI startup, Axiom, is raising $50M at $300 million valuation. [Link](https://techstartups.com/2025/06/04/stanford-math-phd-students-ai-startup-axiom-raises-50m-at-300-million-valuation/)
- CNBC. (2026). MiniMax doubles in Hong Kong debut, marking yet another Chinese AI listing. [Link](https://www.cnbc.com/2026/01/09/minimax-hong-kong-ipo-ai-tigers-zhipu.html)
- Global Times. (2026). AI 'tiger' Zhipu debuts in HK, closing 13% higher than IPO price. [Link](https://www.globaltimes.cn/page/202601/1352704.shtml)
- Nature. (2025). The Chinese finance whizz whose DeepSeek AI model stunned the world (Nature's 10). [Link](https://www.nature.com/articles/d41586-025-03845-4)
- SCMP. (2025). DeepSeek releases first open AI model with gold-level scores at maths olympiad. [Link](https://www.scmp.com/tech/tech-trends/article/3334553/deepseek-releases-first-open-ai-model-gold-level-scores-maths-olympiad)
- CNBC. (2025). Meta acquires intelligent agent firm Manus. [Link](https://www.cnbc.com/2025/12/30/meta-acquires-singapore-ai-agent-firm-manus-china-butterfly-effect-monicai.html)
- TechCrunch. (2026). Meta acquired Moltbook, the AI agent social network. [Link](https://techcrunch.com/2026/03/10/meta-acquired-moltbook-the-ai-agent-social-network-that-went-viral-because-of-fake-posts/)
- SCMP. (2026). MiniMax and Zhipu's stellar Hong Kong IPOs supercharge China's AI ambitions. [Link](https://www.scmp.com/tech/tech-trends/article/3339301/minimax-and-zhipus-stellar-hong-kong-ipos-supercharge-chinas-ai-ambitions)
- Yahoo Tech. (2026). Insiders: DeepSeek V4 beat Claude. [Link](https://tech.yahoo.com/ai/articles/insiders-deepseek-v4-beat-claude-205234497.html)
- Forbes. (2025). Klarna, UPS, Duolingo, Cisco, and many other companies are replacing workers with AI. [Link](https://www.forbes.com/sites/jackkelly/2025/05/04/its-time-to-get-concerned-klarna-ups-duolingo-cisco-and-many-other-companies-are-replacing-workers-with-ai/)
- Wired. (2026). AI agents are hiring humans. [Link](https://www.wired.com/story/ai-agent-rentahuman-bots-hire-humans/)
- BBC. (2026). AI and military applications. [Link](https://www.bbc.com/news/articles/cn48jj3y8ezo)
- FT. (2026). Marrying AI with space engineering. [Link](https://www.ft.com/content/8ee76f65-74d9-4679-a2b0-cd8fc3721a8d)
- Guancha. (2026). Zhang Wenhong opposes AI in hospital diagnostics. [Link](https://www.guancha.cn/politics/2026_01_13_803716.shtml)
- Guancha. (2026). Wang Xiaochuan responds to Zhang Wenhong on medical AI. [Link](https://www.guancha.cn/economy/2026_01_14_803822.shtml)
- China Daily. (2026). Shanghai doctor sparks AI in medicine debate. [Link](https://www.chinadaily.com.cn/a/202601/21/WS69702bf8a310d6866eb34e0d.html)
- GeekPark. (2026). Wang Xiaochuan on AI healthcare. [Link](https://www.geekpark.net/news/344264)
- Stanford HAI. (2025). AI on trial: Legal models hallucinate in 1 out of 6 or more queries. [Link](https://hai.stanford.edu/news/ai-trial-legal-models-hallucinate-1-out-6-or-more-benchmarking-queries)
- CNBC. (2026). Musk's xAI-SpaceX combo is the biggest merger of all time. [Link](https://www.cnbc.com/2026/02/03/musk-xai-spacex-biggest-merger-ever.html)
- TechCrunch. (2026). SpaceX officially acquires xAI. [Link](https://techcrunch.com/2026/02/02/elon-musk-spacex-acquires-xai-data-centers-space-merger/)
- SpaceNews. (2026). SpaceX acquires xAI in bid to develop orbital data centers. [Link](https://spacenews.com/spacex-acquires-xai-in-bid-to-develop-orbital-data-centers/)
- Scientific American. (2026). First Proof is AI's toughest math test yet. The results are mixed. [Link](https://www.scientificamerican.com/article/first-proof-is-ais-toughest-math-test-yet-the-results-are-mixed/)
- Yahoo Finance. (2026). Anthropic's AI safety head just resigned. [Link](https://finance.yahoo.com/news/anthropics-ai-safety-head-just-143105033.html)
- Futurism. (2026). Anthropic researcher quits in cryptic public letter. [Link](https://futurism.com/artificial-intelligence/anthropic-researcher-quits-cryptic-letter)
- CNBC. (2026). Block laying off about 4,000 employees, nearly half of its workforce. [Link](https://www.cnbc.com/2026/02/26/block-laying-off-about-4000-employees-nearly-half-of-its-workforce.html)
- CNN. (2026). Block lays off nearly half its staff because of AI. [Link](https://www.cnn.com/2026/02/26/business/block-layoffs-ai-jack-dorsey)
- Bloomberg. (2026). Jack Dorsey's 4,000 job cuts at Block arouse suspicions of AI-washing. [Link](https://www.bloomberg.com/news/articles/2026-03-01/jack-dorsey-s-4-000-job-cuts-at-block-arouse-suspicions-of-ai-washing)
- CNN. (2026). Anthropic rejects latest Pentagon offer. [Link](https://www.cnn.com/2026/02/26/tech/anthropic-rejects-pentagon-offer)
- CNBC. (2026). Anthropic was the Pentagon's choice for AI. Now it's banned. [Link](https://www.cnbc.com/2026/03/09/anthropic-was-the-pentagons-choice-for-ai-now-its-banned-and-experts-are-worried.html)
- CNBC. (2026). China's tech firms feast on OpenClaw AI agent adoption. [Link](https://www.cnbc.com/2026/03/12/china-openclaw-ai-agent-adoption-tech-companies-government-support-lobster-shrimp.html)
- MIT Technology Review. (2026). Hustlers are cashing in on China's OpenClaw AI craze. [Link](https://www.technologyreview.com/2026/03/11/1134179/china-openclaw-gold-rush/)
- Bloomberg. (2026). China moves to limit use of OpenClaw AI at banks, government agencies. [Link](https://www.bloomberg.com/news/articles/2026-03-11/china-moves-to-limit-use-of-openclaw-ai-at-banks-government-agencies)
- CNBC. (2026). Anthropic's Claude AI and the Pentagon supply chain dispute. [Link](https://www.cnbc.com/2026/03/12/anthropic-claude-emil-michael-defense.html)
- CNBC. (2026). Karp, Palantir weigh in on Anthropic-Pentagon blacklist. [Link](https://www.cnbc.com/2026/03/12/karp-palantir-anthropic-claude-pentagon-blacklist.html)
- New Yorker. (2026). The Pentagon went to war with Anthropic. What's really at stake. [Link](https://www.newyorker.com/news/annals-of-inquiry/the-pentagon-went-to-war-with-anthropic-whats-really-at-stake)
- TechCrunch. (2026). Lawyer behind AI psychosis cases warns of mass casualty risks. [Link](https://techcrunch.com/2026/03/15/lawyer-behind-ai-psychosis-cases-warns-of-mass-casualty-risks/)
- Guardian. (2026). Tumbler Ridge shooting victim sues OpenAI. [Link](https://www.theguardian.com/world/2026/mar/10/tumbler-ridge-shooting-victim-sues-openai-canada)
- China Daily. (2026). AI manipulation of search results highlighted at consumer gala. [Link](http://global.chinadaily.com.cn/a/202603/15/WS69b6be6aa310d6866eb3de9e.html)
- Yicai Global. (2026). China's annual CCTV consumer rights show uncovers AI ad tricks. [Link](https://www.yicaiglobal.com/news/chinas-annual-cctv-consumer-rights-show-uncovers-ai-ad-tricks-that-deceive-customers)
- CNBC. (2026). Iran war and data centers. [Link](https://www.cnbc.com/2026/03/06/iran-war-data-centers.html)
- Fortune. (2026). Iran war, helium shortage, and chip supply chains. [Link](https://fortune.com/2026/03/21/iran-war-helium-shortage-qatar-chip-supply-chains-ai-boom/)
- Bloomberg. (2026). How Amazon data centers became a casualty of the Iran war. [Link](https://www.bloomberg.com/news/articles/2026-03-05/how-amazon-data-centers-became-a-casualty-of-iran-war)
- WSJ. (2026). AI startup Aaru's young founders reach $1B valuation. [Link](https://www.wsj.com/business/ai-startup-aaru-young-founders-35da7f87)
- TechCrunch. (2026). AMI Labs raises $1.03 billion to build world models. [Link](https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/)

---

*Last updated: March 24, 2026*

*These lecture notes are prepared for DOTE 6635 at the CUHK Business School (The Chinese University of Hong Kong). They synthesize fast-moving developments and are intended to provoke discussion, not to provide definitive assessments. Students are encouraged to follow the referenced links and form their own views.*
