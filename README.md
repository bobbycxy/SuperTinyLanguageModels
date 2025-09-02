# Super Tiny Language Models

This GitHub repository presents our research on Super Tiny Language Models (STLMs), aimed at delivering high performance with significantly reduced parameter counts (90-95% smaller) compared to traditional large language models. We explore innovative techniques such as byte-level tokenization with pooling, weight tying, and efficient training strategies. The codebase covers various subproblems, including tokenizer-free models, self-play based training, and alternative training objectives, targeting models with 10M, 50M, and 100M parameters while maintaining competitive performance.

Our mission is to enhance the accessibility and practicality of high-performing language models across a wide range of applications by drastically reducing their computational and energy demands. We believe that our approach has the potential to contribute to more sustainable and inclusive AI development.

For a comprehensive understanding of our research methodology and initial findings, we strongly encourage you to read our paper: [Super Tiny Language Models](https://arxiv.org/pdf/2405.14159)

Please note that this repository is an evolving work in progress, reflecting the ongoing nature of our research. It is subject to frequent updates and improvements as we continue to explore and refine our work on STLMs. We welcome the community's engagement with our work, value your feedback, and appreciate any contributions to this challenging but promising endeavor.

### To Do
- add evals
- add generate text abilities
- add checkpointing and loading methods in STLM object
- add pytests (enforcing the shape of the components at input and output?)
- write over the various components, e.g. FFN, Attn
- improve the train.py. Now is too long. It should be as simple as 
```python
tokenizer = build_tokenizer()

train_dl, val_dl = ...

model = ...

optimizer = ...

trainer = ...

for step in range(max_iterations):
    trainer.train_step(step) # perform optimizer update if step % grad_accum_steps == 0
    
    if step % log_interval == 0:
        trainer.log(step, train_loss)

    if step % eval_interval == 0:
        val_loss = trainer.val_step()
        trainer.log(step, val_loss, split="val")
    
    if step % save_interval == 0:
        trainer.save_checkpoint(step)
```
