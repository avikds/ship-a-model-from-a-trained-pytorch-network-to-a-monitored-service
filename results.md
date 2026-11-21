```
late share: 0.422 | loss per epoch: [0.621, 0.595, 0.596, 0.586, 0.584, 0.581, 0.582, 0.578]
two runs with seed 0 give identical weights: True | test accuracy: 0.697
valid request -> [4.2, 3.0, 18.0, 4.8, 0.7, 1.0] []
bad request -> (None, ['out of range: distance_km', 'not a whole number: items', 'not a number: driver_rating', 'missing: traffic_index', 'unknown field: tip'])
artifact: 8845 bytes | metadata: {'version': '1.0.0', 'features': ['distance_km', 'items', 'hour', 'driver_rating', 'traffic_index', 'rain'], 'test_accuracy': 0.6967}
reloaded model gives identical probabilities: True
TorchScript export, largest difference from the original: 0.0
batched 600 requests in 10 forward passes; same answers as one call: True
8-bit weights: {'float_bytes': 5384, 'quantized_bytes': 1808, 'ratio': 2.98} | accuracy 0.697 -> 0.697
winter drift (PSI): {'distance_km': 0.0152, 'items': 0.0137, 'hour': 0.0068, 'driver_rating': 0.0086, 'traffic_index': 1.9609, 'rain': 0.5676} | drifted: ['traffic_index', 'rain']
shadow run on winter traffic: {'agreement': 0.895, 'labeled': 500, 'prod_accuracy': 0.73, 'cand_accuracy': 0.788, 'decision': 'promote'}
```
