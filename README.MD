# Multi-agents Design Patterns and Evaluation Process

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd multi-agents
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file
cat << EOF > .env
OPENAI_API_KEY=your_openai_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_secret
LANGFUSE_HOST=https://cloud.langfuse.com
EOF
```

## Requirements.txt
```text
langfuse
nltk
python-dotenv
asyncio
pydantic
```


## Design Patterns

### 1. Prompt Chaining
**Name**: Sequential Task Processing  
**Description**: A workflow that breaks down complex tasks into sequential steps where each agent processes the output of the previous one. This pattern is implemented in our project through the Marketing → Validation → Translation chain.

```bash
python multi-agents-design-patterns/prompt-chaining.py
```


### 2. Routing
**Name**: Input Classification and Direction  
**Description**: A pattern that classifies inputs and directs them to specialized handlers. While not directly implemented in our current project, this could be useful for handling different types of marketing content or language pairs.


```bash
python multi-agents-design-patterns/workflow-routing.py
```

### 3. Parallelization
**Name**: Concurrent Task Processing  
**Description**: Enables simultaneous execution of related tasks through either sectioning (breaking into subtasks) or voting (multiple attempts). Our project could be extended to use this for parallel validation checks or multiple translation attempts.

```bash
python multi-agents-design-patterns/parallelization.py
```

### 4. Orchestrator-Workers
**Name**: Dynamic Task Management  
**Description**: Uses a central agent to coordinate and delegate tasks to specialized worker agents. Our project demonstrates this through the main workflow coordination, though in a simplified linear chain.

```bash
python multi-agents-design-patterns/orchestrator-workers.py
```

### 5. Evaluator-Optimizer
**Name**: Iterative Improvement Loop  
**Description**: Implements a feedback loop where one agent generates content and another evaluates and suggests improvements. Our validation agent partially implements this pattern by checking marketing copy quality.

```bash
python multi-agents-design-patterns/evaluator-optimizer.py
```

## Part 2: Evaluation Framework

### Overview
The `eval-agents` folder contains a comprehensive evaluation system for testing and benchmarking translation agents. The system includes parallel translation processing with BLEU score calculation, Langfuse integration for dataset management, and both interactive and batch evaluation modes. The evaluation framework supports English-Spanish translation assessment through multiple agents and uses a voting mechanism to select the best translation among multiple candidates.

### Components
- `translation-agent-eval.py`: Main evaluation script with Langfuse integration
- `translation-agent-local-eval.py`: Local version of the evaluation system
- `build-langfuse-dataset.py`: Tool for creating translation benchmark datasets

### How to Run

1. **Setup**:
```bash

# Set up environment variables in .env file
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_secret
LANGFUSE_HOST=https://cloud.langfuse.com
```

2. **Running Evaluation**:
```bash
python translation-agent-eval.py
```
Then choose:
- Option 1: Interactive mode for single translations
- Option 2: Dataset evaluation mode for batch testing

The system will output translation candidates, BLEU scores, and performance metrics for comprehensive agent evaluation.

3. Run evaluation without LangFuse

```bash
python translation-agent-eval.py
```
