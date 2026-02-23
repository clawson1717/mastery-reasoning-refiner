import torch
import numpy as np
from src.agent import ReasoningAgent

class IterativeGRPO:
    def __init__(self, agent: ReasoningAgent, group_size: int = 4):
        """
        Initializes the Iterative GRPO module.
        
        Args:
            agent: The ReasoningAgent instance to use for generation.
            group_size: Number of samples to generate in each GRPO group.
        """
        self.agent = agent
        self.group_size = group_size

    def score_responses(self, prompt: str, responses: list[str]) -> torch.Tensor:
        """
        Implements a simple relative scoring mechanism.
        
        In Step 7, this will be replaced with a checklist-based reward model.
        Current heuristic: length + presence of keywords + formatting.
        
        Args:
            prompt: The original problem prompt.
            responses: A list of generated reasoning traces.
            
        Returns:
            A torch.Tensor of relative scores (standardized).
        """
        scores = []
        for res in responses:
            score = 0.0
            # Length heuristic (favoring moderately long reasoning)
            # Penalize very short responses, reward substance up to a point
            res_len = len(res)
            if res_len < 50:
                score -= 1.0
            elif 100 < res_len < 800:
                score += 1.0
            
            # Keyword heuristic (presence of logic indicators)
            keywords = ["step", "therefore", "conclude", "because", "since", "analysis"]
            for kw in keywords:
                if kw in res.lower():
                    score += 0.2
            
            # Format heuristic (Markdown bolding or lists often indicate structured thought)
            if "**" in res or "- " in res:
                score += 0.5
                
            scores.append(score)
        
        # Convert to tensor
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        # Relative scoring: standardize within group (GRPO core idea)
        if len(scores_tensor) > 1:
            mean = scores_tensor.mean()
            std = scores_tensor.std()
            if std > 1e-6:
                scores_tensor = (scores_tensor - mean) / std
            else:
                scores_tensor = scores_tensor - mean
        
        return scores_tensor

    def run_refinement_cycle(self, prompt: str, num_iterations: int = 2) -> dict:
        """
        Executes the iterative refinement loop.
        1. Sample a group of traces.
        2. Score them and pick the best.
        3. Iteratively refine the best trace.
        
        Args:
            prompt: The problem to solve.
            num_iterations: Number of refinement steps to take.
            
        Returns:
            A dictionary containing the refinement results and history.
        """
        # 1. Group Sampling & Selection
        initial_responses = self.agent.generate_reasoning(prompt, num_samples=self.group_size)
        initial_scores = self.score_responses(prompt, initial_responses)
        
        best_initial_idx = torch.argmax(initial_scores).item()
        current_trace = initial_responses[best_initial_idx]
        
        history = [
            {"type": "initial_sampling", "trace": current_trace, "score": initial_scores[best_initial_idx].item()}
        ]
        
        # 2. Iterative Refinement
        for i in range(num_iterations):
            refined_trace = self.agent.refine_reasoning(prompt, current_trace)
            
            # Score the refinement (absolute heuristic for tracking)
            # Note: We reuse the scoring logic but don't standardize since it's a single item
            score_val = self.score_responses(prompt, [refined_trace])[0].item()
            
            current_trace = refined_trace
            history.append({
                "type": f"refinement_step_{i+1}",
                "trace": current_trace,
                "score": score_val
            })
            
        return {
            "prompt": prompt,
            "best_initial_trace": initial_responses[best_initial_idx],
            "final_trace": current_trace,
            "history": history
        }

    def train_step_placeholder(self, prompt: str):
        """
        Placeholder for the policy optimization step using GRPO.
        """
        responses = self.agent.generate_reasoning(prompt, num_samples=self.group_size)
        scores = self.score_responses(prompt, responses)
        
        # Placeholder for torch optimization
        # In a real implementation, we would compute log_probs and the GRPO loss:
        # loss = - (log_probs * scores).mean()
        # optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        dummy_loss = torch.tensor(0.0, requires_grad=True)
        # Simulate backprop
        dummy_loss.backward()
        
        return scores
