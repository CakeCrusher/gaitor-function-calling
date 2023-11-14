# Evaluation
Contains the custom evaluation metric, data, and scripts for evaluating the model.

## Custom Evaluation Metric
Let's define:
- \( J_g \) as the generated JSON.
- \( J_e \) as the expected JSON.
- \( K(J) \) as the set of keys in JSON \( J \).
- \( V_k(J) \) as the value corresponding to key \( k \) in JSON \( J \).
- \( Sim_s(a, b) \) as the sentence similarity between strings \( a \) and \( b \).
- \( \text{isDict}(v) \) as a boolean function returning True if \( v \) is a dictionary.
- \( \text{isStr}(v) \) as a boolean function returning True if \( v \) is a string.

For each key \( k \) in \( J_e \):
1. If \( k \in K(J_g) \), set \( key\_sim = 1 \).
2. Else, find \( k' \in K(J_g) \) such that \( k' \) maximizes \( Sim_s(k, k') \) and set \( key\_sim = Sim_s(k, k') \).

For each value \( v \) corresponding to \( k \) in \( J_e \):
1. If \( \text{isDict}(v) \), recursively apply the metric to \( V_k(J_e) \) and \( V_{k'}(J_g) \).
2. Else if \( \text{isStr}(v) \) and \( \text{isStr}(V_{k'}(J_g)) \), set \( value\_sim = Sim_s(v, V_{k'}(J_g)) \).
3. Else if \( v = V_{k'}(J_g) \), set \( value\_sim = 1 \).
4. Else, set \( value\_sim = 0 \).

Calculate the average key similarity \( \text{avg\_key\_sim} \) and the average value similarity \( \text{avg\_value\_sim} \) across all keys.

The final score \( S \) is the average of these two averages:
\[ S = \frac{\text{avg\_key\_sim} + \text{avg\_value\_sim}}{2} \]

This heuristic captures the essence of the custom metric, combining structural comparison with semantic similarity for string values. The recursive aspect is represented by the application of the metric to nested dictionaries. The use of sentence similarity \( Sim_s \) reflects the semantic comparison of string values and keys when an exact match is not found.

## Data
The current evaluation dataset is limited to only function calling outputs.