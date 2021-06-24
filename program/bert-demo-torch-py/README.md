# A question-answering demo program in [PyTorch](http://pytorch.org) based on (HuggingFace transformers)[https://huggingface.co/transformers/]' BERT model and tokenizer

## Should work out of the box using the defaults:

* CK_BERT_MODEL_NAME            : "bert-large-uncased-whole-word-masking-finetuned-squad"
* CK_BERT_DATA_CONTEXT_PATH     : context.txt
* CK_BERT_DATA_QUESTIONS_PATH   : questions.txt
* CK_DEBUG                      : 0

```
$ ck run program:bert-demo-torch-py
```

## Changing the debug level:

See the number-encoded input_ids, token_type_ids and output start..end indices:
```
$ ck run program:bert-demo-torch-py --env.CK_DEBUG=1
```

See the exact vocabulary mapping of both the input and the output:
```
$ ck run program:bert-demo-torch-py --env.CK_DEBUG=2
```

## Changing the model name:

Try [other models](https://huggingface.co/transformers/pretrained_models.html) and see how it affects the result:
```
$ ck run program:bert-demo-torch-py --env.CK_BERT_MODEL_NAME=mrm8488/bert-medium-finetuned-squadv2
```

This may or may not affect the vocabulary mapping:
```
$ ck run program:bert-demo-torch-py --env.CK_BERT_MODEL_NAME=mrm8488/bert-medium-finetuned-squadv2 --env.CK_DEBUG=2
```

## Providing your own context and related questions:

Note: relative filepaths are relative to the program's location
(unlike most CK programs that are relative to program/tmp), but absolute filenames are safe:
```
$ ck run program:bert-demo-torch-py --env.CK_BERT_DATA_CONTEXT_PATH=gagarin_bio.txt --env.CK_BERT_DATA_QUESTIONS_PATH=gagarin_questions.txt
```

