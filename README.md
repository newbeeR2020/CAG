# CAG
Cache-Augmented Generation

## Installation 
```bash
pip install -r ./requirements.txt
```

## Preparation
- create a .env file and input all keys need
```bash
cp ./.env.template ./.env
```

## Parameter Usage -- kvcache.py
- `--kvcache`: "file"
- `--dataset`: "hotpotqa-train" or "squad-train"
- `--similarity` "bertscore"
- `--modelname`: "meta-llama/Llama-3.1-8B-Instruct"
- `--maxKnowledge`: "", int, select how many document in dataset, explanation in Note
- `--maxParagraph`: 100
- `--maxQuestion` int, max question number, explanation in Note
- `--randomSeed`: "", int, a random seed number
- `--output`: "", str, output filepath string
- `--usePrompt`, add this parameter if not using CAG knowledge cache acceleration 

### Example -- kvcache.py
```bash
python ./kvcache.py --kvcache file --dataset "squad-train" --similarity bertscore \
    --maxKnowledge 5 --maxParagraph 100 --maxQuestion 1000  \
    --modelname "meta-llama/Llama-3.1-8B-Instruct" --randomSeed 0 \
    --output "./result_kvcache.txt"
```

## Parameter Usage -- rag.py
- `--index`: "openai" or "bm25"
- `--dataset`: "hotpotqa-train" or "squad-train"
- `--similarity` "bertscore"
- `--maxKnowledge`: "", int, select how many document in dataset, explanation in Note
- `--maxParagraph`: 100
- `--maxQuestion` int, max question number, explanation in Note
- `--topk`: int, the similarity topk of retrieval
- `--modelname`: "meta-llama/Llama-3.1-8B-Instruct"
- `--randomSeed`: "", int, a random seed number
- `--output`: "", str, output filepath string

### Example -- rag.py
```bash
python ./rag.py --index "bm25" --dataset "hotpotqa-train" --similarity bertscore \
    --maxKnowledge 80 --maxParagraph 100 --maxQuestion 80 --topk 3 \
    --modelname "meta-llama/Llama-3.1-8B-Instruct" --randomSeed  0 \
    --output  "./rag_results.txt"
```

### Note:
#### `--maxKnowledge` parameter notice:
> datasets=("squad-train")
> - when k = 3, tokens = 21,000
> - when k = 4, tokens = 32,000
> - when k = 7, tokens = 50,000
> 
> datasets=("hotpotqa-train")
> - all k = 7405 article, tokens = 10,038,084 
> - when k = 1, tokens = 1,400
> - when k = 16, tokens = 22,400
> - when k = 24, tokens = 33,667
> - when k = 32, tokens = 44,800
> - when k = 48, tokens = 64,000
> - when k = 64, tokens = 85,000
> - when k = 80, tokens = 106,000
>
#### `--maxQuestion` parameter notice:
> - when using "hotpotqa-train" dataset, 1 knowledge has 1 question
> - when using "squad-train" dataset, 1 knowledge has average 150 questions
> 


