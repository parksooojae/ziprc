<h1 align="center">ZIP-RC replication</h1>

<p align="center">
  <img src="assets/peony.png" width="800">
</p>



The original ZIP-RC treats each prefix independently when supervising the joint reward + remaining-length prediction. Conceptually these predictions "represent" a value function over prefixes. What if we enforce a Bellman-style constraint during training (e.g. encouraging V(s<sub>t</sub>) to match the expected V(s<sub>t+1</sub>) under the model's own policy) ever come up?

Big step-to-step drops might indicate the model 'knew' a trajectory was failing before that showed up in the value estimate (but it's also possible supervised learning on sequential data already implicitly produces reasonable temporal consistency). By adding temporal consistency constraints, I could potensh get smoother value estimates that make stopping decisions more reliable + better branching heuristics for tree search.

- ZIP-RC: Repurpose unused vocabulary logits for auxiliary predictions to joint distribution over (reward, remaining_length). Uses these predictions to adaptively allocate compute

- ⭐ ZIP-RC-B: Add temporal consistency constraint to enforce V(s<sub>t</sub>) ≈ γ·V(s<sub>t+1</sub>)


