import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

asd = "ASD"

class FunctionCallingMetric:
    def __init__(self, embedder_id="sentence-transformers/bert-base-nli-mean-tokens"):
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedder_id)
        self.embedding_model = AutoModel.from_pretrained(embedder_id)

    def _get_sentence_embedding(self, sentence):
        inputs = self.embedding_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)

        # Mean Pooling - Take attention mask into account for correct averaging
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        return mean_pooled[0].numpy()

    def _sentence_similarity(self, sent1, sent2):
        embedding1 = self._get_sentence_embedding(sent1)
        embedding2 = self._get_sentence_embedding(sent2)
        return 1 - cosine(embedding1, embedding2)

    def run(self, generated_json, expected_json):
        def compare_json(g_json, e_json, key_similarity_scores, value_similarity_scores):
            for e_key, e_value in e_json.items():
                # Check for exact key match or find the most similar key
                if e_key in g_json:
                    g_key = e_key
                    key_similarity_scores.append(1)
                else:
                    # Compute similarity with all keys in generated_json and find the best match
                    key_similarity = {gen_key: self._sentence_similarity(e_key, gen_key) for gen_key in g_json.keys()}
                    g_key, key_sim_score = max(key_similarity.items(), key=lambda x: x[1])
                    key_similarity_scores.append(key_sim_score)

                # Recursive comparison for nested objects, else compare values
                if isinstance(e_value, dict) and isinstance(g_json.get(g_key, {}), dict):
                    compare_json(g_json[g_key], e_value, key_similarity_scores, value_similarity_scores)
                elif isinstance(e_value, str) and isinstance(g_json.get(g_key, ""), str):
                    # Compare values only if they are strings at the root level
                    value_sim_score = self._sentence_similarity(e_value, g_json[g_key])
                    value_similarity_scores.append(value_sim_score)
                elif e_value == g_json.get(g_key, None):
                    value_similarity_scores.append(1)  # Exact match for non-string root values
                else:
                    value_similarity_scores.append(0)  # Non-matching root values

        key_similarity_scores = []
        value_similarity_scores = []
        compare_json(generated_json, expected_json, key_similarity_scores, value_similarity_scores)

        # Combine scores from keys and values as needed, for example, by averaging
        combined_score = sum(key_similarity_scores + value_similarity_scores) / (len(key_similarity_scores) + len(value_similarity_scores))
        return combined_score