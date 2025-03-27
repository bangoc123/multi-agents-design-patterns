# Multi-agents Design Patterns


### 1. Prompt Chaining
**Name**: Sequential Task Processing  
**Description**: A workflow that breaks down complex tasks into sequential steps where each agent processes the output of the previous one. This pattern is implemented in our project through the Marketing → Validation → Translation chain.

### 2. Routing
**Name**: Input Classification and Direction  
**Description**: A pattern that classifies inputs and directs them to specialized handlers. While not directly implemented in our current project, this could be useful for handling different types of marketing content or language pairs.

### 3. Parallelization
**Name**: Concurrent Task Processing  
**Description**: Enables simultaneous execution of related tasks through either sectioning (breaking into subtasks) or voting (multiple attempts). Our project could be extended to use this for parallel validation checks or multiple translation attempts.

### 4. Orchestrator-Workers
**Name**: Dynamic Task Management  
**Description**: Uses a central agent to coordinate and delegate tasks to specialized worker agents. Our project demonstrates this through the main workflow coordination, though in a simplified linear chain.

### 5. Evaluator-Optimizer
**Name**: Iterative Improvement Loop  
**Description**: Implements a feedback loop where one agent generates content and another evaluates and suggests improvements. Our validation agent partially implements this pattern by checking marketing copy quality.



