# Finetune error

## Custom Collate Function

After setting up the environment, I start running `finetune_roma.py` with `finetune.sh`, however I faced the following issue.

```
Traceback (most recent call last):
  File "/root/RoMA/finetune_roma.py", line 301, in <module>
    main()
  File "/root/RoMA/finetune_roma.py", line 203, in main
    for batch in loader:
  File "/root/RoMA/.venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 630, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/root/RoMA/.venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 673, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/RoMA/.venv/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/RoMA/.venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py", line 317, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/RoMA/.venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py", line 192, in collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class '__main__.TrainItem'>
```

Which suggest the default collate function is not able to handle self-defined `TrainItem`.

Therefore, the first modification I made is to add a custom collate function

```python
def custom_collate_fn(batch: List[TrainItem]):
    """
    Custom collate function to handle TrainItem dataclass objects.
    Returns a single object with attributes containing lists.
    """
    class BatchedTrainItem:
        def __init__(self, items):
            self.input_text = [item.input_text for item in items]
            self.correct_answer = [item.correct_answer for item in items]
            self.is_correct = [item.is_correct for item in items]
            self.routing_layers = [item.routing_layers for item in items]
        
        def __iter__(self):
            # Make it iterable to support unpacking if needed
            return iter([self.input_text, self.correct_answer, self.is_correct, self.routing_layers])
        
        def __len__(self):
            return len(self.input_text)
    
    return BatchedTrainItem(batch)
```

And apply the collate function to DataLoaders

```python
    if args.ddp:
        sampler = DistributedSampler(ds, shuffle=True)
-       loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler)
+       loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, collate_fn=custom_collate_fn)
    else:
-       loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler)
+       loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
```

```python
- r_i_list = [r_from_layers_json(it.routing_layers, num_experts, target_layers, device=device) for it in batch]
+ r_i_list = [r_from_layers_json(it, num_experts, target_layers, device=device) for it in batch.routing_layers]
```

## FP32 Gate

After this modification, I face the next issue

```
Traceback (most recent call last):
  File "/root/RoMA/finetune_roma.py", line 322, in <module>
    main()
  File "/root/RoMA/finetune_roma.py", line 291, in main
    scaler.step(optimizer)
  File "/root/RoMA/.venv/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 448, in step
    self.unscale_(optimizer)
  File "/root/RoMA/.venv/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 338, in unscale_
    optimizer_state["found_inf_per_device"] = self._unscale_grads_(
                                              ^^^^^^^^^^^^^^^^^^^^^
  File "/root/RoMA/.venv/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 260, in _unscale_grads_
    raise ValueError("Attempting to unscale FP16 gradients.")
ValueError: Attempting to unscale FP16 gradients.
```

After reading [this issue](https://github.com/huggingface/transformers/issues/23165), I think the easiest way to solve this is simply load the model in float32 instead of float16

```python
- model = OlmoeForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
+ model = OlmoeForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
```

Then we can successfully run the finetune script.