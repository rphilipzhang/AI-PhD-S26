# Introduction

**Course:** DOTE 6635: Artificial Intelligence for Business Research (Spring 2026)
**Instructor:** Renyu (Philip) Zhang

---

## 1. Introduction

These lecture notes are based on the introductory session for the doctoral-level course DOTE 6635: Artificial Intelligence for Business Research, offered in the Spring 2026 semester at the Chinese University of Hong Kong (CUHK). This document provides a comprehensive overview of the course objectives, structure, and the broader landscape of AI applications in business research. The notes are written in an academic article style to facilitate deeper understanding and serve as a reference for students throughout the semester.

---

## 2. Course Evolution: Three Seasons of AI for Business Research

The Spring 2026 semester marks the third iteration of this course, with each "season" exploring different frontiers of artificial intelligence as they relate to business research.

**Season 1 (Spring 2024)** focused on foundational AI technologies, specifically **Natural Language Processing (NLP)** and **Computer Vision (CV)**. These technologies form the backbone of many modern AI applications, from sentiment analysis in marketing to image recognition in retail. The course materials and lecture notes from this season are available on [GitHub](https://github.com/rphilipzhang/AI-PhD-S24) and as a [PDF document](https://rphilipzhang.github.io/rphilipzhang/Scribed_Notes-AI-PhD-S24.pdf).

**Season 2 (Spring 2025)** advanced into the realm of **Large Language Models (LLMs)** and **AI-Powered Causal Inference**. This season reflected the explosive growth of generative AI and its implications for rigorous empirical research. Resources from this season are similarly available on [GitHub](https://github.com/rphilipzhang/AI-PhD-S25) and as [lecture notes](https://rphilipzhang.github.io/rphilipzhang/Scribed_Notes-AI-PhD-S25.pdf).

**Season 3 (Spring 2026)**, the current iteration, ventures into **Reinforcement Learning (RL)** and **Agentic AI**. These topics represent the cutting edge of AI research, where systems learn through interaction with their environment and can act autonomously to achieve complex goals. Approximately 80% of the content in this season is new compared to previous years, with only the introductory sessions on machine learning and deep learning remaining similar.

---

## 3. About the Instructor

Professor Renyu (Philip) Zhang describes himself as "a mostly harmless AI/data science scholar, teacher, and practitioner." His professional roles span academia and industry, serving as a professor at CUHK Business School, an economist at Kuaishou (a major Chinese short-video platform), and the founder of an AI startup. His educational background includes degrees from Peking University (2011) and Washington University in St. Louis (2016), followed by an assistant professorship at NYU Shanghai from 2016 to 2021.

His research agenda centers on a fundamental question: **How can we leverage AI and data science to empower business decision-making, especially for digitalized online platforms?** This question motivates much of the course content and provides a unifying theme for the various AI applications discussed throughout the semester.

---

## 4. The Thinking Game and the Power of AI Agents

The lecture opened with a reference to "The Thinking Game," a concept associated with Sir Demis Hassabis, co-founder of DeepMind and a pioneer in artificial intelligence research. This framing sets the stage for understanding AI as a tool for augmenting human cognitive capabilities.

### 4.1 Demonstration of Coding Agents

A compelling demonstration of modern AI capabilities was presented through the example of **coding agents**. Professor Zhang described a replication exercise he uses to assess new students and research assistants: replicating Figure 7 from one of his research papers on deep learning. Traditionally, this task would require significant programming expertise and time. However, with the advent of AI coding agents such as [OpenAI Codex](https://openai.com/codex/), this task can now be accomplished with minimal human intervention. The demonstration showed how a coding agent could understand the requirements of the replication task, write the necessary code to reproduce the figure, and execute the code to generate the output. This example illustrates a broader trend in AI: the emergence of systems that can autonomously complete complex, multi-step tasks that previously required human expertise. These "agentic" AI systems represent a paradigm shift in how we interact with computers and will be a major focus of this course.

---

## 5. The Bitter Lesson

A foundational concept introduced in this lecture is **"The Bitter Lesson,"** articulated by Professor Richard Sutton, a pioneering figure in reinforcement learning research. This lesson encapsulates decades of experience in AI development and offers crucial guidance for researchers.

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."

The "bitter" aspect of this lesson lies in its implications for researchers. While incorporating domain-specific knowledge into AI systems can yield short-term gains and is often intellectually satisfying, such approaches tend to plateau over time. In contrast, general methods that can scale with increasing computational resources have consistently proven more effective in the long run. This principle has profound implications for business research. Domain-specific approaches may provide quick wins but can inhibit further progress, so research that scales with data and compute is more likely to have lasting impact. General methods, while potentially less elegant, tend to outperform specialized approaches as resources increase. The course will emphasize research approaches that align with this principle, focusing on methods that can leverage the exponential growth in computational power and data availability.

---

## 6. Defining AI for Business Research

The course defines **AI for Business Research** as the application of AI technologies to address, or facilitate the addressing of, business research questions. This definition encompasses a wide range of applications, from using machine learning to analyze consumer behavior to employing deep learning for demand forecasting.

### 6.1 The AI Flywheel

A central concept in understanding AI's role in business is the **AI Flywheel**, a virtuous cycle that drives continuous improvement:

| Component               | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| **Business Applications** | Real-world use cases that generate value and data     |
| **Data**                | The fuel for training and improving AI models         |
| **AI Models**             | Algorithms that learn patterns and make predictions   |
| **Compute**               | Computational resources that enable model training and inference |

This flywheel effect creates a self-reinforcing cycle: better AI models lead to better business applications, which generate more data, which in turn enables even better AI models. The course emphasizes that researchers should focus on work that can scale with data and compute, aligning with the principles of the Bitter Lesson.

---

## 7. Course Purpose and Objectives

The primary purpose of this course is to provide students with a foundational understanding of the key concepts and methods in machine learning (ML) and artificial intelligence (AI) that are relevant to business research. The course aims to develop a basic understanding of fundamental ML and AI concepts and methods used in business research. It also seeks to help students understand how business researchers have utilized ML/AI and what managerial questions have been addressed. Finally, the course will cultivate a sense of what state-of-the-art AI/ML technologies can accomplish, both in the ML/AI community and potentially in students' own research fields.

### 7.1 The Coverage vs. Depth Trade-off

A fundamental challenge in designing this course is the trade-off between breadth and depth of coverage. The course takes a deliberate approach by providing a concise introduction to AI/ML topics relevant to applied business research. It covers enough necessary knowledge to help students understand key trade-offs and potentially invent new applied methods. The course also informs students about literature development in relevant domains and prepares them with the necessary sense to conduct rigorous business research using these methods. A key emphasis is placed on the combination of coding and theory for practical implementation.

### 7.2 Distinguishing CS and Business Research Impact

The course draws an important distinction between the factors that drive impact in computer science versus business research:

| Field                | Impact Formula                                                 |
| -------------------- | -------------------------------------------------------------- |
| **Computer Science** | Problem Importance × Technical Novelty × Performance Improvement |
| **Business Research**  | Problem Importance × Identification Rigor × Insight Novelty    |

This distinction helps students understand how to position their research and what aspects to emphasize when applying AI methods to business questions.

---

## 8. Prerequisites and Expectations

### 8.1 Prior Knowledge Assumptions

The course assumes students have a working knowledge in calculus, linear algebra, and statistics. A working knowledge of Python programming is also expected, though AI coding assistants like Codex, Claude Code, and Cursor can now help with this. Finally, some basic sense of ML, causal inference, and econometrics is preferred but not strictly required.

### 8.2 Important Warnings

The course comes with several important caveats. At CUHK, there is an Economics course (ECON 5180) covering similar topics to Season 1 without the coding emphasis. This may be the most time-consuming course at CUHK by a wide margin. While ideas and methods will be discussed in class with demos, students will need coding skills (potentially "vibe coding" with AI assistance) to complete homework and replication projects.

---

## 9. Course Format and Structure

### 9.1 Weekly Session Format

Each weekly session spans 2 hours and 45 minutes, structured as follows:

| Duration    | Activity                                          |
| ----------- | ------------------------------------------------- |
| 15 minutes  | Homework discussions and review of previous content |
| 105 minutes | Theories and coding demos                         |
| 30 minutes  | Student presentations                             |

### 9.2 Group Work

All coursework is completed in groups of at most **two students**. Students were required to register their group members, majors, and group name by 11:59 PM on January 7, 2026. Unregistered students would be matched with others based on their majors. Each student must evaluate their partner's contribution to all coursework.

---

## 10. Coursework and Grading

The course includes four main components of coursework. Each group will be responsible for scribing the lecture notes for one session/topic throughout the semester. In addition, each group will replicate and present one paper per week. There will be weekly coding assignments, due two weeks after distribution, with the best five assignments counting toward the final grade. Finally, each group will complete a research project based on their own choice. All homework and the final project must be completed in **Python**. Detailed grading information is available in the [course syllabus](https://github.com/rphilipzhang/AI-PhD-S26/blob/main/AI-PhD-Syllabus-S2026.pdf).

---

## 11. Course Materials and Resources

### 11.1 Primary Resources

| Resource                                       | URL                                                                                             |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| GitHub Repository                              | https://github.com/rphilipzhang/AI-PhD-S26                                                        |
| Google Sheet (Registration, Sign-ups, Submissions) | https://docs.google.com/spreadsheets/d/1YIwCR-X8VVLTv-OfGr7DqZJF6YIll-JE3SGI1q7JEus/edit?usp=sharing |
| Google CoLab (Code Demos)                      | https://drive.google.com/drive/folders/19806VHq6Vybrx-z4BbvV01sTMs9G5Hrh?usp=sharing          |

### 11.2 Course Communications

| Channel       | Details                                                  |
| ------------- | -------------------------------------------------------- |
| Class Meeting | Tuesday, 9:30 AM – 12:15 PM @ CYT 209A (CYT LT5 on March 3) |
| Office Hour   | By appointment @ CYT_911                                 |
| WeChat Group  | Online discussion forum (200+ members)                   |
| Instructor Email | philipzhang@cuhk.edu.hk                                  |
| Teaching Assistant | Ignis Jiang (ignisjiang@cuhk.edu.hk)                     |

### 11.3 Python Tutorial Sessions

Two optional Python tutorial sessions are offered online on Friday evenings (7:00 PM – 9:00 PM), taught by Xinyu Li, an MIS PhD Candidate at CUHK Business School. The first session on January 9, 2026, will cover Python Basics. The second session on January 16, 2026, will cover PyTorch Basics & FBA/DOT Server.

---

## 12. Alternative Resources for Learning AI

For students seeking additional learning resources, the course recommends several excellent options:

| Topic                       | Instructor      | Resource                                                              |
| --------------------------- | --------------- | --------------------------------------------------------------------- |
| Basic ML Introduction       | Andrew Ng       | [Coursera](https://www.coursera.org/specializations/machine-learning-introduction) |
| Deep Learning Introduction  | Andrew Ng       | [Coursera](https://www.coursera.org/specializations/deep-learning)      |
| Natural Language Processing | Chris Manning   | [Stanford CS224n](https://web.stanford.edu/class/cs224n/)               |
| Computer Vision             | Fei-Fei Li      | [Stanford CS231n](http://cs231n.stanford.edu/)                         |
| Deep Reinforcement Learning | Sergey Levine   | [UC Berkeley](https://rail.eecs.berkeley.edu/deeprlcourse/)             |
| Deep Learning Theory        | Matus Telgarsky | [UIUC](https://mjt.cs.illinois.edu/courses/dlt-f22/)                    |
| Machine Learning Fairness   | Moritz Hardt    | [Fairness and Machine Learning](https://fairmlbook.org/)                |
| Large Language Models       | Danqi Chen      | [Princeton COS 597G](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/) |
| Generative AI Short Courses | Various         | [DeepLearning.AI](https://www.deeplearning.ai/short-courses/)           |

---

## 13. What is AI/ML?

Machine learning is defined as a **computer science subfield that automates computers to learn from data without being explicitly programmed**. This field goes by various names depending on the context and community, including Data Mining, Statistical Learning, and Data Science. The course will use these terms somewhat interchangeably while acknowledging their subtle differences in emphasis and methodology.

---

## 14. The Landscape of AI/ML for Business Research

A significant portion of the introductory lecture was devoted to mapping the landscape of how AI and ML are used in business research. This landscape can be organized into several distinct categories, each representing a different way that AI technologies intersect with business research questions.

### 14.1 A Typical Applied Business Research Paper

The structure of a typical applied business research paper can be understood through the lens of AI applications:

| Stage                         | Traditional Approach    | AI Enhancement                         |
| ----------------------------- | ----------------------- | -------------------------------------- |
| Setting & Problems            | Manual identification   | ML as Data Source                      |
| Data Generation               | Survey, observational   | ML for data extraction                 |
| Prediction/Optimization Models | Econometric models      | ML for Predictive Decision-Making      |
| Hypothesis Testing            | Statistical tests       | ML for Causal Inference                |
| Mechanisms                    | Theoretical analysis    | ML for Structural Estimation           |
| Counterfactuals               | Structural models       | OR for LLM Inference                   |
| Decisions                     | Policy recommendations  | Generative AI for Content/Decision     |

### 14.2 AI/ML as Data Source

One of the most fundamental applications of AI in business research is using machine learning to transform unstructured data into analyzable information. Any recordable information that is not inherently numerical can potentially be analyzed with ML to answer business questions. This includes text (Natural Language Processing), images and video (Computer Vision), sound (Deep Learning), and even genetic information (Bioinformatics). Generative AI has significantly expanded the scope of analyzable information, enabling researchers to extract insights from data sources that were previously inaccessible. The benefits of using ML to understand unstructured data include cost reduction, scalability, objectivity, and easy integration into other systems. However, this approach also presents challenges related to measurement errors, interpretation, and potential biases in training data.

### 14.3 Issues with ML as Data Source

When using ML-generated data in empirical models, researchers must carefully consider how measurement errors can affect their analyses. The key parameter of interest in any empirical model may be biased if the outcome, treatment, or control variables are generated through ML with error. For instance, if the treatment variable is generated with an error that is correlated with the outcome, this can lead to biased causal estimates. Similarly, errors in ML-generated control variables can affect the validity of the analysis. In these contexts, double machine learning can be applied for effective debiasing.

### 14.4 LLM for Social Simulations

A cutting-edge application of AI in business research involves using Large Language Models for social simulations. Recent research has explored using LLMs to simulate human research subjects, providing an accessible data source for understanding human behavior. This includes building "digital twins" of individuals based on their survey responses and addressing challenges related to diversity, bias, sycophancy, alienness, and generalization in LLM simulations. Representative papers in this area include work on improving behavioral alignment in LLM social simulations and creating datasets for building digital twins of over 2,000 people.

### 14.5 AI/ML for Causal Inference

Machine learning techniques are increasingly being integrated with causal inference methods to improve the rigor and scalability of empirical research. Key resources in this area include the [CausalML Book](https://causalml-book.org/), the [Stanford GSB SI Lab ML-CI Tutorial](https://bookdown.org/stanfordgsbsilab/ml-ci-tutorial/), and previous course materials from [AI-PhD-S25](https://github.com/rphilipzhang/AI-PhD-S25). Representative research includes work on deep learning-based causal inference for large-scale combinatorial experiments, influencer selection using follower elasticity, and targeting for long-term outcomes.

### 14.6 AI/ML for Predictive Decision-Making and Optimization

This category represents the classic application of machine learning in business contexts. Examples include comparing customer choice models with machine learning for finding optimal product displays on platforms like Alibaba, using dynamic coupon targeting with batch deep reinforcement learning for livestream shopping, and developing cold start algorithms to improve market thickness on online advertising platforms. These applications demonstrate how ML can be used not just for prediction but for optimizing complex business decisions in real-time.

### 14.7 LLM for Operations Research

The intersection of Large Language Models and Operations Research represents an emerging frontier. Research in this area includes the development of ORLM, a customizable framework for training large models for automated optimization modeling, and CALM before the STORM, which aims to unlock native reasoning for optimization modeling using Large Reasoning Models (LRMs). These approaches aim to leverage the reasoning capabilities of LLMs to solve complex optimization problems that traditionally required specialized expertise.

### 14.8 Generative AI for Content and Decision

Generative AI is being applied to create content and inform decisions in various business contexts. This includes applying LLMs to generate ad text optimized for click-through rates in search engine advertising, developing unified generative models like OneRec for retrieval and ranking with iterative preference alignment in recommendation systems, and creating end-to-end generative recommendation and reinforcement learning-based bidding models for marketing services.

### 14.9 ML as Subject

AI and ML have themselves become subjects of academic study. Researchers are examining the economics of AI, machine-human collaboration, ML fairness and discrimination, the impact of ML on the labor market, data privacy, and the role of data and ML in Industrial Organization. Some are even considering AI as a new kind of species. Representative research includes experimental evidence on the productivity effects of generative AI and empirical studies of algorithmic bias in online advertising.

### 14.10 ML for Structural Estimation

Machine learning is being used to estimate parameters of structural models, which are widely used in economics and marketing. Key approaches include adversarial estimation, which uses generative adversarial networks for structural estimation, and neural network estimators, which train neural nets to provide parameter estimates for structural econometric models. These methods can offer computational advantages and robustness to model misspecification.

### 14.11 OR for LLM Inference

Operations Research techniques are being applied to optimize the inference process for Large Language Models. This includes developing fluid-guided online scheduling with memory constraints for LLM inference optimization and fundamental modeling for LLM inference with exploding KV cache demands. These approaches address the practical challenges of deploying LLMs at scale, including memory management and scheduling efficiency.

---

## 15. Tentative Course Schedule

The course is organized into four main modules:

| Module                        | Sessions |
| ----------------------------- | -------- |
| Introduction to Supervised Learning | 1        |
| Introduction to Deep Learning | 1        |
| Reinforcement Learning        | 6        |
| Agentic AI                    | 5        |

The schedule is tentative and subject to changes. Students should refer to the syllabus and GitHub repository for the most up-to-date information.

---

## 16. Conclusion

This introductory lecture has established the foundation for the course "DOTE 6635: Artificial Intelligence for Business Research." The course aims to equip doctoral students with the knowledge and skills necessary to engage with the rapidly evolving field of AI and to apply these powerful tools to their own research. Key takeaways from this introduction include the importance of the Bitter Lesson and focusing on methods that scale with data and compute, the AI Flywheel as a framework for understanding AI's role in business, the diverse landscape of AI applications in business research, and the emergence of Agentic AI as a transformative paradigm for autonomous problem-solving. By combining theoretical understanding with practical coding skills, students will be prepared to contribute to the next generation of AI-powered business research.

---

## References

[1] Sutton, R. (2019). The Bitter Lesson. http://www.incompleteideas.net/IncIdeas/BitterLesson.html

[2] Zhang, R. (2024). AI-PhD-S24 GitHub Repository. https://github.com/rphilipzhang/AI-PhD-S24

[3] Zhang, R. (2024). Scribed Notes AI-PhD-S24. https://rphilipzhang.github.io/rphilipzhang/Scribed_Notes-AI-PhD-S24.pdf

[4] Zhang, R. (2025). AI-PhD-S25 GitHub Repository. https://github.com/rphilipzhang/AI-PhD-S25

[5] Zhang, R. (2025). Scribed Notes AI-PhD-S25. https://rphilipzhang.github.io/rphilipzhang/Scribed_Notes-AI-PhD-S25.pdf

[6] Zhang, R. (2026). AI-PhD-S26 GitHub Repository. https://github.com/rphilipzhang/AI-PhD-S26

[7] Hassabis, D. The Thinking Game. https://www.bilibili.com/video/BV1bbU8BREog/?t=1468s

[8] Zhang, R. DeDL-nonblind.pdf. https://rphilipzhang.github.io/rphilipzhang/DeDL-nonblind.pdf

[9] OpenAI. Codex. https://openai.com/codex/

[10] Sutton, R. The Bitter Lesson [Video]. https://www.youtube.com/watch?v=vbVfAqPI8ng

[11] Ng, A. Machine Learning Introduction. https://www.coursera.org/specializations/machine-learning-introduction

[12] Ng, A. Deep Learning Specialization. https://www.coursera.org/specializations/deep-learning

[13] Manning, C. CS224n: Natural Language Processing with Deep Learning. https://web.stanford.edu/class/cs224n/

[14] Li, F.-F. CS231n: Deep Learning for Computer Vision. http://cs231n.stanford.edu/

[15] Levine, S. Deep Reinforcement Learning. https://rail.eecs.berkeley.edu/deeprlcourse/

[16] Telgarsky, M. (2022). Deep Learning Theory. https://mjt.cs.illinois.edu/courses/dlt-f22/

[17] Hardt, M., & Recht, B. Fairness and Machine Learning. https://fairmlbook.org/

[18] Chen, D. (2022). COS 597G: Understanding Large Language Models. https://www.cs.princeton.edu/courses/archive/fall22/cos597G/

[19] DeepLearning.AI. Short Courses on Generative AI. https://www.deeplearning.ai/short-courses/

[20] Zhang, R. (2026). AI-PhD-Syllabus-S2026. https://github.com/rphilipzhang/AI-PhD-S26/blob/main/AI-PhD-Syllabus-S2026.pdf

[21] CausalML Book. https://causalml-book.org/

[22] Stanford GSB SI Lab. Machine Learning for Causal Inference Tutorial. https://bookdown.org/stanfordgsbsilab/ml-ci-tutorial/

[23] Kong, L., Jin, J., & Zhang, R. (2026). Improving Behavioral Alignment in LLM Social Simulations via Context Formation and Navigation. Working paper.

[24] Ye, Z., Zhang, Z., Zhang, D. J., Zhang, H., & Zhang, R. (2023). Deep Learning Based Causal Inference for Large-Scale Combinatorial Experiments: Theory and Empirical Evidence. *Management Science*.

[25] Ye, Z., Zhang, D. J., Zhang, H., Zhang, R., Chen, X., & Xu, Z. (2023). Cold start to improve market thickness on online advertising platforms: Data-driven algorithms and field experiments. *Management Science*, 69(7), 3838-3860.

[26] Zhang, X., Sun, C., Zhang, R., & Goh, K-Y. (2024). The Value of AI-Generated Metadata for UGC Platforms: Evidence from a Large-scale Field Experiment. Working paper and CIST 2024.

[27] Noy, S., & Zhang, W. (2023). Experimental evidence on the productivity effects of generative artificial intelligence. *Science*, 381(6654), 187-192.

[28] Lambrecht, A., & Tucker, C. (2019). Algorithmic Bias? An Empirical Study of Apparent Gender-Based Discrimination in the Display of STEM Career Ads. *Management Science*, 65(7), 2966-2981.

[29] Kaji, T., Manresa, E., & Pouliot, G. (2023). An Adversarial Approach to Structural Estimation. *Econometrica*, 91(6), 2041-2063.

[30] Wei, Y., & Jiang, Z. (2024). Estimating Parameters of Structural Models Using Neural Networks. *Marketing Science*.
