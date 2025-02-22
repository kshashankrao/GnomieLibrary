## Self Attention

Self attention defines the relationship of a single element of the sequence to the rest of the sequence. The math behind it is as follows:

1. Let the sentence be "The dog chased the cat". Here "chased" is an important word as its context is dependent on both dog and cat. 
2. Transform the words into embeddings --> x_0 (for "The"), x_1 (for "dog") ...
3. Initialize the weights for Query w_q, Key q_k and Value q_v. 
4. The input embeddings are transferred to Q, K and V. 

    $Qx_0 = w_q * x_0, Qx_1 = w_q * x_1, ...$

    $Kx_0 = w_k * x_0, Kx_1 = w_k * x_1, ...$

    $Vx_0 = w_v * x_0, Vx_1 = w_v * x_1, ...$
   
6. Calculate the similarity score 

   $score\_Qx_0 = Qx_0 * Kx_0$

   $scores = [score\_Qx_0, ...]$
   
8. Take the softmax to get the probabilties and then mutliply it with the value. This results in attention scores. 

   $Attention\_score = softmax(scores) * [Vx_0, ...]$
   

 
