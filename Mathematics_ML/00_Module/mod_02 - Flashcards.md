---
id: "mod_02_flashcards"
title: "Module - Flashcards"
page_type: flashcards_page
module_id: "mod_02"
card_count: 227
source_count: 177
---

- Total cards: 227
- Total source objects: 177

## Flashcards

> [!question]- 1. What is the central problem addressed in the Statistical Learning chapter?
> The central problem is choosing a hypothesis $h$ that minimizes the generalization risk $R(h)$ when the underlying joint distribution $P_0$ is unknown.
>> [!info]- Source
>> [[obj_006 - 1 Statistical learning]]

> [!question]- 2. How is the Bayes classifier formally defined?
> The Bayes classifier is $h^* := \text{argmin}_h R(h)$, where $R(h)$ is the misclassification risk.
>> [!info]- Source
>> [[obj_011 - 1.2 The Bayes classifier]]

> [!question]- 3. What is the 'Tower property' of conditional expectation?
> It states that $\mathbb{E}(\mathbb{E}(Z \mid W) \mid f(W)) = \mathbb{E}(Z \mid f(W))$. A simpler version is $\mathbb{E}(\mathbb{E}(Z \mid W)) = \mathbb{E}(Z)$.
>> [!info]- Source
>> [[obj_010 - 1.2.1 Conditional expectation]]

> [!question]- 4. State the Conditional Jensen's inequality.
> For a convex function $f$, $\mathbb{E}(f(Z) \mid W) \geq f(\mathbb{E}(Z \mid W))$.
>> [!info]- Source
>> [[obj_010 - 1.2.1 Conditional expectation]]

> [!question]- 5. What is the 'regression function' $\eta(x)$?
> It is the conditional probability $\mathbb{P}(Y = 1 \mid X = x)$.
>> [!info]- Source
>> [[obj_014 - 1.2.2 Characterization of the Bayes classifier]]

> [!question]- 6. Define the empirical risk $\hat{R}(h)$ for a dataset of size $n$.
> $\hat{R}(h) = \frac{1}{n} \sum_{i=1}^n \ell(h(X_i), Y_i)$.
>> [!info]- Source
>> [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]

> [!question]- 7. What is an 'Empirical Risk Minimizer' (ERM)?
> The ERM is $\hat{h} \in \text{argmin}_{h \in \mathcal{H}} \hat{R}(h)$, the function in the class $\mathcal{H}$ that achieves the lowest training error.
>> [!info]- Source
>> [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]

> [!question]- 8. What happens to variance and bias as the complexity of the hypothesis class $\mathcal{H}$ increases?
> As complexity increases, the model can fit the training data better (decreasing bias) but becomes more sensitive to specific training samples (increasing variance).
>> [!info]- Source
>> [[obj_022 - 1.4 Bias-variance trade-off]]

> [!question]- 9. In the error decomposition, what does the 'noise' term represent?
> It represents the irreducible error $\mathbb{E}((h^*(X) - Y)^2)$ due to the inherent stochasticity of the target variable $Y$.
>> [!info]- Source
>> [[obj_022 - 1.4 Bias-variance trade-off]]

> [!question]- 10. How does the variance of a K-NN estimator depend on $K$?
> The variance is given by $\sigma^2/K$, meaning it decreases as the number of neighbors $K$ increases.
>> [!info]- Source
>> [[obj_023 - 1.4.1 K nearest neighbours]]

> [!question]- 11. Why does bias increase for large $K$ in K-NN?
> As $K$ increases, the estimator averages over a larger region, making it more like the global average and less responsive to the local variations of the regression function $h^*$.
>> [!info]- Source
>> [[obj_023 - 1.4.1 K nearest neighbours]]

> [!question]- 12. What is the natural global objective cross-validation attempts to optimize?
> It attempts to minimize the expected generalization error $\mathcal{R}(k) = \mathbb{E}_{D \sim P^n}(\mathbb{E}_{(X,Y) \sim P} (\ell(h_{k,D}(X), Y) \mid D))$.
>> [!info]- Source
>> [[obj_024 - 1.4.2 Cross-validation]]

> [!question]- 13. Why is V-fold cross-validation preferred over hold-out validation?
> V-fold cross-validation reduces the variance of the error estimate by averaging over multiple splits, ensuring every data point is used for both training and validation.
>> [!info]- Source
>> [[obj_024 - 1.4.2 Cross-validation]]

> [!question]- 14. What are the two key questions addressed in the Excess Risk section?
> 1. How does the complexity of $\mathcal{H}$ influence excess risk? 2. How does the number of data points $n$ influence excess risk?
>> [!info]- Source
>> [[obj_028 - 1.5 Excess risk]]

> [!question]- 15. State the excess risk decomposition formula.
> $R(\hat{h}) - R(h^*) = (R(\hat{h}) - \inf_{h \in \mathcal{H}} R(h)) + (\inf_{h \in \mathcal{H}} R(h) - R(h^*))$, representing estimation and approximation error respectively.
>> [!info]- Source
>> [[obj_029 - 1.5.1 Decomposition of excess risk]]

> [!question]- 16. What is the goal of the 'Estimation Error' section?
> To quantify how the gap between empirical risk and true risk depends on the number of samples $n$ and the size/complexity of the hypothesis class $\mathcal{H}$.
>> [!info]- Source
>> [[obj_031 - 1.6 The estimation error]]

> [!question]- 17. What is the difference between an asymptotic and a non-asymptotic result in this context?
> Asymptotic results (like CLT) describe behavior as $n \to \infty$, whereas non-asymptotic results (like Hoeffding's) provide explicit bounds that are valid for specific, finite $n$.
>> [!info]- Source
>> [[obj_035 - 1.6.1 Tools from probability]]

> [!question]- 18. What is the primary limitation of the theory in this subsection?
> It only applies to finite hypothesis classes, whereas many important machine learning models (like linear models or neural networks) are infinite.
>> [!info]- Source
>> [[obj_048 - 1.6.2 Finite hypothesis classes]]

> [!question]- 19. What does Rademacher complexity intuitively measure?
> It measures a function class's ability to 'fit' random noise; if it can fit noise well, it is highly complex and prone to overfitting.
>> [!info]- Source
>> [[obj_051 - 1.6.3 Rademacher complexity]]

> [!question]- 20. What is the core idea behind using VC dimension to bound risk?
> The core idea is to count the number of distinct behaviors $|F(z_{1:n})|$ that a function class can produce on a dataset of size $n$, which then bounds the Rademacher complexity via Massart's Lemma.
>> [!info]- Source
>> [[obj_061 - 1.6.4 Vapnik-Chernovenkis (VC) dimension]]

> [!question]- 21. How does VC dimension relate to the growth function?
> The VC dimension is the largest integer $n$ such that the growth function $\Pi_{\mathcal{H}}(n) = 2^n$, meaning the class can still shatter $n$ points.
>> [!info]- Source
>> [[obj_061 - 1.6.4 Vapnik-Chernovenkis (VC) dimension]]

> [!question]- 22. What are the three main sub-themes of this optimization section?
> 1. Convexity and optimality conditions; 2. Constrained optimization and Duality; 3. Iterative methods (Gradient Descent and SGD).
>> [!info]- Source
>> [[obj_069 - 2 Optimization]]

> [!question]- 23. What is the goal of the optimization problems discussed here?
> The goal is to devise efficient iterative algorithms to approximate a minimizer $w^* \in \text{argmin } f(w)$ with low computational cost per iteration.
>> [!info]- Source
>> [[obj_068 - 2.1 Preliminaries]]

> [!question]- 24. Why is convexity so desirable in machine learning?
> Because for convex functions, every local minimizer is also a global minimizer, meaning simple descent algorithms will not get stuck in suboptimal 'valleys'.
>> [!info]- Source
>> [[obj_071 - 2.1.1 Convexity]]

> [!question]- 25. What defines the feasibility set $\mathcal{F}$ in standard form?
> It is the set of all $w$ satisfying $f_i(w) \leq 0$ for all $i$ and $Aw = b$.
>> [!info]- Source
>> [[obj_085 - 2.2 Constrained optimization]]

> [!question]- 26. What is the difference between weak and strong duality?
> Weak duality is the property that the dual optimal value is always less than or equal to the primal optimal value, whereas strong duality is the condition where these two values are exactly equal.
>> [!info]- Source
>> [[obj_086 - 2.2.1 Duality]]

> [!question]- 27. What role does the Lagrange function play in duality?
> The Lagrange function combines the objective function and the constraints into a single expression using Lagrange multipliers, allowing the dual function to be defined as the infimum of this expression over the primal variables.
>> [!info]- Source
>> [[obj_086 - 2.2.1 Duality]]

> [!question]- 28. What is the 'kernel trick' in SVMs?
> It is the method of replacing inner products of data points with a kernel function $k(x, x')$, allowing the algorithm to operate in high-dimensional feature spaces without explicitly computing feature vectors.
>> [!info]- Source
>> [[obj_099 - 2.3 Support vector machines]]

> [!question]- 29. What is the formula for the distance from $x$ to the hyperplane $w^\top u + b = 0$?
> The distance is $\frac{|w^\top x + b|}{\|w\|}$.
>> [!info]- Source
>> [[obj_098 - 2.3.1 Distance to the hyperplane]]

> [!question]- 30. How is the primal SVM optimization problem defined?
> It is defined as $\text{argmin}_{w, b} \|w\|$ s.t. $Y_i(w^\top X_i + b) \ge 1$ for all $i = 1, \dots, n$.
>> [!info]- Source
>> [[obj_100 - 2.3.2 Primal formulation of SVM]]

> [!question]- 31. What are 'support vectors' in the context of the dual SVM problem?
> Support vectors are the data points for which the corresponding dual multiplier $\xi_i$ is non-zero; these points lie exactly on the margin boundary.
>> [!info]- Source
>> [[obj_101 - 2.3.3 The dual problem for SVM]]

> [!question]- 32. What is the role of the parameter $\mu$ in soft-margin SVM?
> The parameter $\mu$ (sometimes called $C$) controls the trade-off between maximizing the margin and minimizing the total training error (misfit).
>> [!info]- Source
>> [[obj_102 - 2.3.4 Non-exact separation]]

> [!question]- 33. What is the purpose of the feature map $\Phi(x)$?
> The feature map transforms data into a high-dimensional space where a linear separator may exist for data that is nonlinearly separable in the original space.
>> [!info]- Source
>> [[obj_104 - 2.3.5 Nonlinear decision boundaries]]

> [!question]- 34. Why is gradient descent necessary?
> Because for many complex objectives (like those in deep learning), we cannot solve the optimality condition $\nabla f = 0$ in closed form.
>> [!info]- Source
>> [[obj_110 - 2.4 Gradient descent]]

> [!question]- 35. What is the specific form of the objective function analyzed in the quadratic convergence section?
> The objective function is assumed to be a quadratic form $f(w) = \frac{1}{2}\langle Cw, w \rangle - \langle w, b \rangle$, where $C$ is symmetric and positive definite.
>> [!info]- Source
>> [[obj_112 - 2.4.1 Convergence analysis for gradient descent]]

> [!question]- 36. What determines the convergence speed of gradient descent on a quadratic objective?
> The speed is determined by the eigenvalues of the Hessian matrix $C$, specifically the ratio between the minimum and maximum eigenvalues.
>> [!info]- Source
>> [[obj_112 - 2.4.1 Convergence analysis for gradient descent]]

> [!question]- 37. Why do we define strong convexity and smoothness in the context of optimization?
> We define them to provide global bounds on how much the function and its gradient can change, which allows us to guarantee convergence rates for iterative algorithms like gradient descent.
>> [!info]- Source
>> [[obj_118 - 2.4.2 Strong convexity and smoothness]]

> [!question]- 38. What does Lipschitz smoothness tell us about the gradient of a function?
> It tells us that the gradient cannot change arbitrarily fast, meaning the surface of the function is 'locally' somewhat predictable and doesn't have sharp 'kinks' or extreme spikes in curvature.
>> [!info]- Source
>> [[obj_118 - 2.4.2 Strong convexity and smoothness]]

> [!question]- 39. How do convergence rates for gradient descent change as we add more assumptions like strong convexity?
> More assumptions generally lead to faster theoretical rates, moving from $O(1/\sqrt{k})$ for just Lipschitz, to $O(1/k)$ for smooth, to linear for strongly convex and smooth functions.
>> [!info]- Source
>> [[obj_122 - 2.4.3 Gradient descent under strong convexity]]

> [!question]- 40. What is the 'Idea' behind SGD as presented in Subsection 2.5?
> The idea is to replace the expensive full gradient computation with a gradient of a single component function $\nabla f_{i_k}(w)$ where $i_k$ is sampled uniformly at random.
>> [!info]- Source
>> [[obj_127 - 2.5 Stochastic gradient descent]]

> [!question]- 41. What property of the stochastic gradient ensures the algorithm can still converge to the correct minimum?
> The stochastic gradient is an unbiased estimator of the true gradient, meaning $\mathbb{E}(\nabla f_{i_k}(w)) = \nabla f(w)$.
>> [!info]- Source
>> [[obj_127 - 2.5 Stochastic gradient descent]]

> [!question]- 42. How is a single-hidden-layer neural network mathematically defined?
> It is defined as a function $h_{w,a,b}(x) = \sum_{i=1}^q a_i \sigma(x^\top w_i + b_i)$, where $\sigma$ is a nonlinear activation function.
>> [!info]- Source
>> [[obj_133 - 3.1 Multilayer perceptrons]]

> [!question]- 43. What are some popular choices for the activation function $\sigma$?
> Common choices include the sigmoid function $\sigma(r) = e^r / (1 + e^r)$ and the Rectified Linear Unit (ReLU) $\rho(r) = \max(r, 0)$.
>> [!info]- Source
>> [[obj_133 - 3.1 Multilayer perceptrons]]

> [!question]- 44. How is a deep neural network defined recursively?
> It is defined by $F^1(x) = \sigma(W_1x + b^1)$ and $F^{k+1}(x) = \sigma(W_{k+1}F^k(x) + b^{k+1})$ for $k = 1, ..., L-1$.
>> [!info]- Source
>> [[obj_135 - 3.1.1 Deep neural networks]]

> [!question]- 45. What is the 'format' of a deep neural network?
> The format is a vector $d = (d_1, ..., d_L)$ specifying the number of neurons in each of the $L$ layers.
>> [!info]- Source
>> [[obj_135 - 3.1.1 Deep neural networks]]

> [!question]- 46. Who are the key researchers credited with the early Universal Approximation results?
> Cybenko (using the Hahn-Banach theorem) and Hornik (using the Stone-Weierstrass theorem) in the late 1980s.
>> [!info]- Source
>> [[obj_138 - 3.1.2 Universal approximation]]

> [!question]- 47. How is the Kullback-Leibler divergence defined for discrete distributions $P$ and $Q$?
> $KL(P|Q) = \sum_{\omega \in \Omega} \log (P(\omega)/Q(\omega)) P(\omega)$.
>> [!info]- Source
>> [[obj_141 - 3.2 Cross-entropy loss]]

> [!question]- 48. What is the relationship between minimizing KL divergence and Maximum Likelihood Estimation (MLE)?
> Finding $\theta$ to minimize $KL(P|Q_\theta)$ is equivalent to maximizing the expected log-likelihood $\mathbb{E}_P(\log(Q_\theta))$.
>> [!info]- Source
>> [[obj_141 - 3.2 Cross-entropy loss]]

> [!question]- 49. What is 'back-propagation' in the context of automatic differentiation?
> It is a clever implementation of the chain rule and a special case of reverse mode automatic differentiation applied to neural networks.
>> [!info]- Source
>> [[obj_143 - 3.3 Automatic differentiation]]

> [!question]- 50. What is the main drawback of symbolic differentiation?
> The resulting derivative expressions can become extremely large and memory-intensive to store and evaluate.
>> [!info]- Source
>> [[obj_142 - 3.3.1 Three approaches to (numerical) differentiation]]

> [!question]- 51. Why is finite differences often unsuitable for training large neural networks?
> It only provides an approximation of the gradient and requires $p+1$ evaluations of the function for $p$ parameters, which is computationally expensive.
>> [!info]- Source
>> [[obj_142 - 3.3.1 Three approaches to (numerical) differentiation]]

> [!question]- 52. What does an edge from node $z_k$ to node $z_l$ indicate in a computational graph?
> It indicates that the intermediate value $z_k$ is used as an input to compute the function $f_l$ associated with node $z_l$.
>> [!info]- Source
>> [[obj_144 - 3.3.2 Computational graphs]]

> [!question]- 53. How are initial derivative values $\dot{z}_{-i}$ initialized in forward mode to compute $\partial f / \partial x_j$?
> They are initialized as $1$ if $i = j$ and $0$ otherwise.
>> [!info]- Source
>> [[obj_145 - 3.3.3 Forward mode]]

> [!question]- 54. What value is $\dot{z}_{n+1}$ initialized to in reverse mode?
> It is initialized to 1, representing $\partial f / \partial z_{n+1}$ where $f = z_{n+1}$.
>> [!info]- Source
>> [[obj_151 - 3.3.4 Reverse mode]]

> [!question]- 55. State the recursive formula for $\dot{z}_k$ in reverse mode.
> $\dot{z}_k = \sum_{m \in \text{Child}(k)} \dot{z}_m \frac{\partial z_m}{\partial z_k}$.
>> [!info]- Source
>> [[obj_151 - 3.3.4 Reverse mode]]

> [!question]- 56. Which architecture family is particularly important for Natural Language Processing (NLP)?
> Transformers, introduced in 2017.
>> [!info]- Source
>> [[obj_159 - 3.4 Common neural network architectures]]

> [!question]- 57. Why do CNNs have fewer parameters than fully connected networks for large images?
> Because they use weight sharing: the same small convolutional filter is applied to every spatial position in the input, whereas a fully connected layer would require separate weights for every pixel interaction.
>> [!info]- Source
>> [[obj_158 - 3.4.1 Convolutional neural networks (CNN)]]

> [!question]- 58. How is translation invariance achieved in a CNN?
> By using convolution operators, which are the only linear operators that are invariant to translations.
>> [!info]- Source
>> [[obj_158 - 3.4.1 Convolutional neural networks (CNN)]]

> [!question]- 59. How are tokens typically enhanced with spatial information in a Transformer?
> Each token $x_i$ might contain positional information, such as a number indicating its relative position, appended as the last entry in its vector.
>> [!info]- Source
>> [[obj_163 - 3.4.2 Transformers]]

> [!question]- 60. What are the two main components of a GAN and their respective goals?
> The Generator $G$ aims to generate data indistinguishable from the true distribution $\rho_X$. The Discriminator $D$ aims to estimate the probability that a sample came from the real data rather than the generator.
>> [!info]- Source
>> [[obj_165 - 3.5 Generative adversarial networks]]

> [!question]- 61. State the min-max objective function $V(D, G)$ for GANs.
> $\min_G \max_D V(D, G)$ where $V(D, G) = \mathbb{E}_{X \sim \rho_X}(\log(D(X))) + \mathbb{E}_{Z \sim \rho_Z}(\log(1 - D(G(Z))))$.
>> [!info]- Source
>> [[obj_165 - 3.5 Generative adversarial networks]]

> [!question]- 62. How does a VAE differ from a standard autoencoder?
> A standard autoencoder uses deterministic mappings for encoding and decoding, whereas a VAE models these as conditional distributions $q_\theta(z|x)$ and $p_\theta(x|z)$.
>> [!info]- Source
>> [[obj_170 - 3.6 Variational autoencoders]]

> [!question]- 63. List three reasons why dimension reduction is desirable.
> To visualize data more easily, to remove noise, and to lower resource requirements for storage/processing.
>> [!info]- Source
>> [[obj_169 - 3.6.1 Dimension reduction]]

> [!question]- 64. How is a sample in the data space generated in a VAE?
> First, draw a sample $z \sim p_Z$ from the latent prior, then sample from the conditional distribution $p_\theta(\cdot|z)$.
>> [!info]- Source
>> [[obj_174 - 3.6.2 Variational autoencoders]]

> [!question]- 65. Why is the Central Limit Theorem insufficient for bounding estimation error in this context?
> The CLT is an asymptotic result (as $n \to \infty$) and does not provide quantitative bounds for finite $n$ that hold uniformly over a hypothesis class.
>> [!info]- Source
>> [[obj_030 - A naive approach]]

> [!question]- 66. What are the two main steps in a single iteration of the GAN training algorithm?
> 1. Fix the generator and update the discriminator using $N$ steps of SGD to maximize the value function. 2. Fix the discriminator and update the generator using gradient descent to minimize the value function.
>> [!info]- Source
>> [[obj_168 - Algorithm for GAN optimization]]

> [!question]- 67. What is the optimization problem for training a basic autoencoder?
> $\min_{E,D} \frac{1}{n} \sum_{i=1}^n ||D(E(x_i)) - x_i||^2$.
>> [!info]- Source
>> [[obj_171 - Autoencoders]]

> [!question]- 68. What are the recursive variables in the backpropagation equations?
> The layer-wise sensitivities $\dot{z}_i^m = \sigma'(z_i^m)\dot{a}_i^m$ and $\dot{a}_i^{m-1} = \sum_k \dot{z}_k^m W_{k,i}^m$.
>> [!info]- Source
>> [[obj_155 - Backpropagation]]

> [!question]- 69. What are the main components of the neural networks chapter?
> The chapter covers MLPs and expressiveness, loss functions (cross-entropy), automatic differentiation (backpropagation), common architectures (CNNs, Transformers), and generative models (GANs, VAEs).
>> [!info]- Source
>> [[obj_134 - Chapter 3 Neural networks]]

> [!question]- 70. What is the Armijo rule?
> The Armijo rule (or backtracking line search) is a strategy to choose a stepsize that guarantees a sufficient decrease in the function value by checking the inequality $f(w_k + \tau d_k) \le f(w_k) + \alpha \tau \langle d_k, \nabla f(w_k) \rangle$.
>> [!info]- Source
>> [[obj_109 - Choosing the stepsize]]

> [!question]- 71. Why can't we use standard backpropagation directly through the sampling step $z \sim q_\psi$?
> Standard backpropagation requires deterministic differentiable nodes; the stochastic nature of sampling prevents gradients from flowing back to the parameters $\psi$ of the encoder.
>> [!info]- Source
>> [[obj_181 - Computation of gradients]]

> [!question]- 72. What are the three main chapters covered in these lecture notes?
> The three main chapters are (1) Statistical Learning, (2) Optimization, and (3) Neural Networks.
>> [!info]- Source
>> [[obj_001 - Contents]]

> [!question]- 73. Which section covers the bias-variance trade-off?
> The bias-variance trade-off is discussed in section 1.4, within the Statistical Learning chapter.
>> [!info]- Source
>> [[obj_001 - Contents]]

> [!question]- 74. What are the three defining properties of a Linear Time-Invariant (LTI) operator?
> 1. Linearity: $L(\alpha f) = \alpha L(f)$. 2. Additivity: $L(f + g) = L(f) + L(g)$. 3. Shift-invariance: $L(f(\cdot - \tau)) = (L(f))(\cdot - \tau)$.
>> [!info]- Source
>> [[obj_160 - Convolutions and Fourier transforms in signal processing]]

> [!question]- 75. What is the 'filter' in the context of an LTI operator?
> The filter $h$ is defined as the operator's response to a Dirac measure, i.e., $h = L(\delta)$.
>> [!info]- Source
>> [[obj_160 - Convolutions and Fourier transforms in signal processing]]

> [!question]- 76. What are the requirements for using Hoeffding's inequality?
> The random variables must be independent and bounded within known intervals $(a_i, b_i)$.
>> [!info]- Source
>> [[obj_044 - Corollary 6 (Hoeffding’s inequality)]]

> [!question]- 77. How does the bound in Hoeffding's inequality change with the number of samples $n$?
> The tail probability decreases exponentially as $n$ increases, since $n$ appears in the negative exponent.
>> [!info]- Source
>> [[obj_044 - Corollary 6 (Hoeffding’s inequality)]]

> [!question]- 78. Write the formula for excess risk $\mathcal{E}$.
> $\mathcal{E} := R(\hat{h}) - R(h^*)$.
>> [!info]- Source
>> [[obj_025 - Definition 1]]

> [!question]- 79. What is the mathematical definition of $\mu$-strong convexity?
> A differentiable function $f$ is $\mu$-strongly convex if for all $x, x'$, $\langle \nabla f(x) - \nabla f(x'), x - x' \rangle \ge \mu \|x - x'\|^2$.
>> [!info]- Source
>> [[obj_116 - Definition 10 (Strong convexity and smoothness)]]

> [!question]- 80. What is the mathematical definition of $L$-Lipschitz smoothness?
> A function $f$ is $L$-Lipschitz smooth if its gradient satisfies $\|\nabla f(x) - \nabla f(x')\| \le L \|x - x'\|$ for all $x, x'$.
>> [!info]- Source
>> [[obj_116 - Definition 10 (Strong convexity and smoothness)]]

> [!question]- 81. Give the mathematical expression for the convolution of $f$ and $h$.
> $g(t) := (f \star h)(t) := \int f(x)h(t - x)dx$.
>> [!info]- Source
>> [[obj_156 - Definition 11 (Convolution)]]

> [!question]- 82. Define the discrete circular convolution of sequences $h$ and $f$.
> $g_k = \sum_{i=1}^n f_{k-i}h_i$.
>> [!info]- Source
>> [[obj_157 - Definition 12 (Discrete convolution)]]

> [!question]- 83. State the MGF condition for a random variable $W$ to be sub-Gaussian with parameter $\sigma$.
> $\mathbb{E}(e^{\alpha(W-\mathbb{E}(W))}) \le e^{\alpha^2\sigma^2/2}$ for all $\alpha \in \mathbb{R}$.
>> [!info]- Source
>> [[obj_036 - Definition 2]]

> [!question]- 84. What is the difference between empirical and population Rademacher complexity?
> Empirical complexity is fixed for a given dataset $z_{1:n}$, while population complexity is the expectation over the data $\mathcal{R}_n(\mathcal{F}) = \mathbb{E}(\hat{\mathcal{R}}(\mathcal{F}(Z_{1:n})))$.
>> [!info]- Source
>> [[obj_052 - Definition 3 (Rademacher complexity)]]

> [!question]- 85. What does it mean for a hypothesis class to 'shatter' a set of points?
> A set of points is shattered if the hypothesis class can achieve every possible binary labeling $(2^n)$ on those points.
>> [!info]- Source
>> [[obj_062 - Definition 4]]

> [!question]- 86. Is the VC dimension always finite?
> No, for very complex classes (like the set of all possible classifiers), the VC dimension can be infinite.
>> [!info]- Source
>> [[obj_062 - Definition 4]]

> [!question]- 87. Can a function have a local minimizer but no global minimizer?
> Yes, for example $f(w) = -w^2$ has no global minimizer as it goes to $-\infty$, or $f(w) = \exp(-w)$ which has an infimum but no attainable global minimizer.
>> [!info]- Source
>> [[obj_070 - Definition 5]]

> [!question]- 88. What is the difference between convexity and strict convexity?
> Convexity allows for 'flat' regions (linear segments), while strict convexity requires the function to be strictly 'curved' upwards, ensuring a unique minimum.
>> [!info]- Source
>> [[obj_072 - Definition 6]]

> [!question]- 89. What are the Lagrange multipliers in the Lagrange function?
> They are vectors $\xi$ and $\nu$ that weight the constraints $Aw - b$ and $f_k(w)$ in the combined objective.
>> [!info]- Source
>> [[obj_084 - Definition 7 (Duality)]]

> [!question]- 90. How is the Lagrange dual function $D(\xi, \nu)$ calculated?
> $D(\xi, \nu) = \inf_{w \in \mathbb{R}^p} L(w, \xi, \nu)$.
>> [!info]- Source
>> [[obj_084 - Definition 7 (Duality)]]

> [!question]- 91. How is the dual optimization problem mathematically formulated?
> It is formulated as $\sup_{\xi \in \mathbb{R}^n, \nu \in \mathbb{R}^m} D(\xi, \nu)$ subject to the constraint $\nu \ge 0$.
>> [!info]- Source
>> [[obj_087 - Definition 8 (The dual problem)]]

> [!question]- 92. What is the mathematical condition for strong duality?
> Strong duality holds if $\sup_{\xi \in \mathbb{R}^n, \nu \in \mathbb{R}^m_{\ge 0}} D(\xi, \nu) = \inf_{w \in \mathcal{F}} f_0(w)$.
>> [!info]- Source
>> [[obj_088 - Definition 9 (Weak and strong duality)]]

> [!question]- 93. What is the result of multiplying $(a + b\varepsilon)$ and $(c + d\varepsilon)$?
> $ac + (ad + bc)\varepsilon$, since the $bd\varepsilon^2$ term is zero.
>> [!info]- Source
>> [[obj_149 - Dual numbers]]

> [!question]- 94. How is the application of a function $f$ to a dual number defined?
> $f(a + b\varepsilon) := f(a) + f'(a)b\varepsilon$.
>> [!info]- Source
>> [[obj_149 - Dual numbers]]

> [!question]- 95. In the autoencoder framework, what determines the optimal matrix $Q$ for PCA?
> The optimal $Q$ consists of the $d$ largest orthonormal eigenvectors of $XX^\top$.
>> [!info]- Source
>> [[obj_172 - Example (Principle component analysis)]]

> [!question]- 96. According to the example, what is the cost of one iteration of SGD compared to GD for $n$ data points in $p$ dimensions?
> One iteration of SGD costs $O(p)$, whereas one iteration of full GD costs $O(np)$, making SGD $n$ times cheaper.
>> [!info]- Source
>> [[obj_124 - Example (SGD)]]

> [!question]- 97. In the Gaussian mixture model, how is $\eta(x)$ computed using class densities $\rho_0, \rho_1$?
> By Bayes' rule, $\eta(x) = \frac{q\rho_1(x)}{(1-q)\rho_0(x) + q\rho_1(x)}$, where $q = \mathbb{P}(Y=1)$.
>> [!info]- Source
>> [[obj_017 - Example 1]]

> [!question]- 98. What is the VC dimension of a hypothesis class consisting of only two constant functions, $h_1 = 1$ and $h_2 = -1$?
> The VC dimension is 1, because it can shatter a single point (achieving both labels) but cannot shatter two points (which would require 4 labelings).
>> [!info]- Source
>> [[obj_063 - Example 10]]

> [!question]- 99. Why can't the interval class shatter 3 points on a line?
> If three points are ordered $x_1 < x_2 < x_3$, an interval cannot contain $x_1$ and $x_3$ without also containing $x_2$, making the labeling (1, 0, 1) impossible.
>> [!info]- Source
>> [[obj_065 - Example 11]]

> [!question]- 100. What is the VC dimension of axis-aligned rectangles in 2D?
> The VC dimension is 4.
>> [!info]- Source
>> [[obj_066 - Example 12]]

> [!question]- 101. What are the normal equations for least squares?
> $A^\top A w^* = A^\top b$.
>> [!info]- Source
>> [[obj_077 - Example 13 (Least squares)]]

> [!question]- 102. How is a norm constraint $\|w\| \leq K$ expressed in standard optimization form?
> It is expressed as an inequality constraint $f_1(w) = \sum_{i} w_i^2 - K^2 \leq 0$.
>> [!info]- Source
>> [[obj_083 - Example 14]]

> [!question]- 103. Why does strong duality fail in Example 15?
> Strong duality fails because there is no 'gap' in the feasible region that allows the dual lower bound to reach the primal minimum; specifically, the problem lacks a point in the interior of the domain that satisfies the constraints strictly.
>> [!info]- Source
>> [[obj_089 - Example 15]]

> [!question]- 104. What is the dual problem for the least squares problem $\min \|x\|^2$ s.t. $Ax = b$?
> The dual problem is $\max_z -1/4 \|A^\top z\|^2 + \langle b, z \rangle$.
>> [!info]- Source
>> [[obj_092 - Example 16]]

> [!question]- 105. In the context of Example 17, how is the greedy stepsize $\tau_k$ calculated for the least squares objective?
> The stepsize is calculated using the formula $\tau_k = \|r_k\|^2 / \|Ar_k\|^2$, where $r_k = A^\top(Aw_k - b)$ is the gradient.
>> [!info]- Source
>> [[obj_111 - Example 17 (Least squares)]]

> [!question]- 106. What is the geometric implication of choosing the greedy stepsize in this example?
> Choosing the greedy stepsize ensures that consecutive search directions are orthogonal ($\langle r_k, r_{k+1} \rangle = 0$), which often leads to a zig-zag trajectory toward the minimizer.
>> [!info]- Source
>> [[obj_111 - Example 17 (Least squares)]]

> [!question]- 107. Why can't the XOR function be solved by a linear separator?
> Because the points (0,1) and (1,0) cannot be separated from (0,0) and (1,1) by a single line in 2D space.
>> [!info]- Source
>> [[obj_136 - Example 18]]

> [!question]- 108. What activation function is used in the network constructed in Example 18?
> The ReLU activation function, denoted as $(x)_+ = \max(0, x)$.
>> [!info]- Source
>> [[obj_136 - Example 18]]

> [!question]- 109. What is the closed-form solution for the parameters $(\hat{w}, \hat{b})$ in linear regression ERM?
> $\begin{pmatrix} \hat{w} \\ \hat{b} \end{pmatrix} = (M^\top M)^{-1} M^\top Y_{1:n}$, where $M$ is the feature matrix with an added column of ones.
>> [!info]- Source
>> [[obj_019 - Example 2]]

> [!question]- 110. What is the difference between 1-NN and K-NN in terms of noise sensitivity?
> 1-NN is highly sensitive to noise or mislabeled examples because its prediction depends on a single training point; K-NN reduces this sensitivity by averaging over $K$ points.
>> [!info]- Source
>> [[obj_021 - Example 3]]

> [!question]- 111. How is the hypothesis class $\mathcal{H}$ defined for a 1-NN classifier?
> It consists of indicator functions on Voronoi cells, where each cell contains points closer to a specific training example than to any other.
>> [!info]- Source
>> [[obj_021 - Example 3]]

> [!question]- 112. In the learning rectangles example, what is the required sample size $n$ to achieve risk $\epsilon$ with confidence $1-\delta$?
> $n \ge \frac{4}{\epsilon} \log(4/\delta)$.
>> [!info]- Source
>> [[obj_027 - Example 4]]

> [!question]- 113. What value of $\alpha$ achieves the infimum in the Chernoff bound for a $\mathcal{N}(0, \sigma^2)$ variable?
> The infimum is achieved at $\alpha = t/\sigma^2$.
>> [!info]- Source
>> [[obj_032 - Example 5]]

> [!question]- 114. What is the MGF of a Rademacher random variable?
> $\mathbb{E}(e^{\alpha \varepsilon}) = \frac{1}{2}(e^\alpha + e^{-\alpha}) = \cosh(\alpha)$.
>> [!info]- Source
>> [[obj_037 - Example 6]]

> [!question]- 115. For a grid of $m^2$ squares, what is the size of the hypothesis class $|\mathcal{H}|$?
> Since there are $m^2$ squares and each has 2 possible labels, $|\mathcal{H}| = 2^{m^2}$.
>> [!info]- Source
>> [[obj_049 - Example 7]]

> [!question]- 116. Why is the Rademacher complexity of a singleton set zero?
> Because for a fixed function $f$, the expectation $\mathbb{E}_{\varepsilon}(\sum \varepsilon_i f(z_i))$ sums to zero since $\mathbb{E}(\varepsilon_i) = 0$.
>> [!info]- Source
>> [[obj_053 - Example 8]]

> [!question]- 117. What is the Rademacher complexity $\mathcal{R}_n(\mathcal{H})$ for linear models $w^\top \phi(x)$ with $\|w\| \le K$?
> $\mathcal{R}_n(\mathcal{H}) \le \frac{K \sqrt{\mathbb{E}(\|\phi(X)\|^2)}}{\sqrt{n}}$.
>> [!info]- Source
>> [[obj_057 - Example 9 (Linear models)]]

> [!question]- 118. What value is computed simultaneously with $f(2,3)$ in this example?
> The directional derivative $(r_1, r_2)\nabla f(2,3)$.
>> [!info]- Source
>> [[obj_150 - Example of Dual Numbers]]

> [!question]- 119. How is the forward pass used in reverse mode AD?
> It is used to compute and store the intermediate values (like $z_3 = \log(2)$) which are needed to evaluate the partial derivatives $\partial z_m / \partial z_k$ during the backward pass.
>> [!info]- Source
>> [[obj_152 - Example of Reverse Mode]]

> [!question]- 120. Why can't $\sigma$ be a linear function if we want high expressivity?
> Because if $\sigma$ is linear, the composition $h_{w,a,b}(x)$ is still a linear function in $x$, meaning the network cannot learn non-linear patterns.
>> [!info]- Source
>> [[obj_137 - Expressiveness]]

> [!question]- 121. What happens if the activation function $\sigma$ is a polynomial of degree $d$?
> The resulting network $h_{w,a,b}(x)$ will always be a polynomial of degree $d$, limiting its capacity to approximate non-polynomial functions.
>> [!info]- Source
>> [[obj_137 - Expressiveness]]

> [!question]- 122. State the update rule for the gradient descent algorithm.
> The update rule is $w_{k+1} = w_k - \tau_k \nabla f(w_k)$.
>> [!info]- Source
>> [[obj_108 - Gradient descent algorithm]]

> [!question]- 123. Why must the stepsize $\tau_k$ in SGD eventually converge to 0?
> It must converge to 0 to cancel out the 'noise' induced by stochastic sampling from individual data points.
>> [!info]- Source
>> [[obj_128 - How to choose the stepsize ̄̄̄̄̄̄̄̄̄̄]]

> [!question]- 124. What is a common stepsize schedule used for SGD as described in the notes?
> A common schedule is $\tau_k = \tau_0 / (1 + k/k_0)$, where $k_0$ serves as a warm-up phase parameter.
>> [!info]- Source
>> [[obj_128 - How to choose the stepsize ̄̄̄̄̄̄̄̄̄̄]]

> [!question]- 125. What effect does the term $-\frac{1}{2} ||F_\psi(x)||^2$ have on the latent space?
> It penalizes large latent codes, encouraging the representation to stay close to the origin in the latent space.
>> [!info]- Source
>> [[obj_178 - Interpretation of ELBO terms]]

> [!question]- 126. Define the Gaussian (RBF) kernel.
> $k(x, x') = \exp(-\|x - x'\|^2 / (2\sigma^2))$.
>> [!info]- Source
>> [[obj_105 - Kernels]]

> [!question]- 127. Why is the kernel formulation efficient?
> Because the dual SVM problem only depends on inner products, replacing them with a kernel allows us to skip the expensive or impossible step of computing high-dimensional feature vectors.
>> [!info]- Source
>> [[obj_105 - Kernels]]

> [!question]- 128. What does Massart's Lemma relate?
> It relates empirical Rademacher complexity to the number of distinct behaviors $|\mathcal{F}(z_{1:n})|$ of a function class on a dataset.
>> [!info]- Source
>> [[obj_058 - Lemma 10. [Massart’s Lemma]]]

> [!question]- 129. What is the bound on the expected excess risk provided by Sauer-Shelah?
> $\mathbb{E}(R(\hat{h}) - \inf_{h \in \mathcal{H}} R(h)) \leq 2 \sqrt{\frac{2 VC(\mathcal{H}) \log(n+1)}{n}}$.
>> [!info]- Source
>> [[obj_067 - Lemma 12 (Sauer-Shelah Lemma)]]

> [!question]- 130. Is the dual function always concave?
> Yes, the dual function is always concave because it is the pointwise infimum of affine functions of the multipliers, even if the primal objective is not convex.
>> [!info]- Source
>> [[obj_090 - Lemma 17]]

> [!question]- 131. State Markov's inequality for a non-negative random variable $W$.
> $\mathbb{P}(W \ge t) \le \frac{\mathbb{E}(W)}{t}$ for all $t > 0$.
>> [!info]- Source
>> [[obj_033 - Markov's inequality]]

> [!question]- 132. How does max-pooling help neurons in deeper layers of a CNN?
> By reducing the spatial resolution, max-pooling allows neurons in deeper layers to be dependent on larger and larger areas of the original image domain, capturing more elaborate shapes.
>> [!info]- Source
>> [[obj_161 - Max-pooling]]

> [!question]- 133. What function is typically used to model the parameters of the Gaussian conditional $p_\theta(\cdot|z)$?
> A deterministic function $G_\theta$ (a neural network) is used to map the latent variable $z$ to the mean of the data distribution.
>> [!info]- Source
>> [[obj_180 - Maximizing the ELBO]]

> [!question]- 134. Why is the direct MLE objective $p_\theta(x) = \int p_\theta(x|z)p_Z(z)dz$ difficult to optimize?
> It is intractable because we need to compute an integral over all possible latent states $z$, which requires evaluating the mapping at every point.
>> [!info]- Source
>> [[obj_173 - Maximum likelihood estimation]]

> [!question]- 135. How is the stochastic gradient with respect to $\psi$ evaluated after reparameterization?
> Since $z$ is now a deterministic function of $\psi$, the gradient $\partial_\psi$ can be evaluated directly via backpropagation.
>> [!info]- Source
>> [[obj_183 - Monte-carlo approximations]]

> [!question]- 136. How should one choose the parameters $d$ or $\sigma$ for SVM kernels?
> These parameters should generally be selected using cross-validation to balance the trade-off between model fit and generalization error.
>> [!info]- Source
>> [[obj_106 - Practical considerations]]

> [!question]- 137. What previously proved results are used in the proof of Hoeffding's Inequality?
> Hoeffding's Lemma (Lemma 4) to establish sub-Gaussianity of bounded variables, Proposition 5 for the parameter of the sum, and Proposition 3 for the final tail bound.
>> [!info]- Source
>> [[obj_045 - Proof of Corollary 6]]

> [!question]- 138. What sub-Gaussian parameter is used for the Rademacher averages in this proof?
> The parameter is $n^{-1/2}$.
>> [!info]- Source
>> [[obj_059 - Proof of Lemma 10]]

> [!question]- 139. How does the proof establish the lower bound property?
> By showing that for any feasible $w$, the terms $\xi^\top (Aw - b)$ and $\sum \nu_k f_k(w)$ in the Lagrangian are zero and non-positive respectively, making $L(w, \xi, \nu) \le f_0(w)$.
>> [!info]- Source
>> [[obj_091 - Proof of Lemma 17]]

> [!question]- 140. What property of KL divergence allows simplifying the multivariate Gaussian case?
> The additivity of KL divergence under product measures, given that coordinates are independent.
>> [!info]- Source
>> [[obj_182 - Proof of Lemma 27]]

> [!question]- 141. What is the role of the Rademacher random variable in the proof of Hoeffding's Lemma?
> It is used in the symmetrization step to represent the distribution of the difference $W - W'$, allowing the expectation to be bounded by the sub-Gaussian property of Rademacher variables.
>> [!info]- Source
>> [[obj_041 - Proof of Lemma 4]]

> [!question]- 142. Why is Jensen's conditional inequality applied in this proof?
> It is used to bound the moment generating function $’\mathbb{E}(e^{\alpha W})’$ by the expectation of the exponential of the difference $’\mathbb{E}(e^{\alpha(W-W')})’$.
>> [!info]- Source
>> [[obj_041 - Proof of Lemma 4]]

> [!question]- 143. What is the first step in the proof of Proposition 1?
> The first step is to use the law of total expectation to write $R(h) = \mathbb{E}(\mathbb{E}(\mathbf{1}(h(X) \neq Y) \mid X))$.
>> [!info]- Source
>> [[obj_015 - Proof of Proposition 1]]

> [!question]- 144. How does the first-order convexity characterization prove sufficiency of $\nabla f = 0$?
> By substituting $\nabla f(w^*) = 0$ into $f(w) \geq f(w^*) + \langle \nabla f(w^*), w - w^* \rangle$, we immediately obtain $f(w) \geq f(w^*)$.
>> [!info]- Source
>> [[obj_078 - Proof of Proposition 15]]

> [!question]- 145. Why is the directional derivative non-negative at a constrained minimum?
> If it were negative, moving a small distance $\lambda$ in that direction (which stays in the convex feasible set) would yield a lower function value.
>> [!info]- Source
>> [[obj_079 - Proof of Proposition 16]]

> [!question]- 146. How is the error update represented in the proof of Proposition 20?
> The error at step $k+1$ is related to step $k$ via the operator $(Id - \tau_k C)$, such that $w_{k+1} - w^* = (Id - \tau_k C)(w_k - w^*)$.
>> [!info]- Source
>> [[obj_113 - Proof of Proposition 20]]

> [!question]- 147. How does the stepsize $\tau_k$ affect the contraction factor in this proof?
> The contraction factor $\rho$ is determined by $\|Id - \tau_k C\|$. Choosing $\tau_k$ such that this norm is strictly less than 1 ensures convergence.
>> [!info]- Source
>> [[obj_113 - Proof of Proposition 20]]

> [!question]- 148. How is the fundamental theorem of calculus used in the proof of Proposition 21?
> It is used to express the difference $f(y) - f(x)$ as an integral of the directional derivative $h'(t) = \langle \nabla f(x + t(y - x)), y - x \rangle$ from $t=0$ to $t=1$.
>> [!info]- Source
>> [[obj_119 - Proof of Proposition 21]]

> [!question]- 149. What is the key substitution made in the smoothness part of the proof?
> The term $\langle \nabla f(x + t(y-x)) - \nabla f(x), y-x \rangle$ is bounded by $tL\|y-x\|^2$ using Cauchy-Schwarz and the Lipschitz smoothness property.
>> [!info]- Source
>> [[obj_119 - Proof of Proposition 21]]

> [!question]- 150. Which property of logarithms is utilized in the proof of Proposition 26?
> The concavity of the logarithm function is used to apply Jensen's inequality.
>> [!info]- Source
>> [[obj_177 - Proof of Proposition 26]]

> [!question]- 151. What property of independent variables is central to the proof of Proposition 5?
> The fact that for independent variables $X_i$, the expectation of the product $\mathbb{E}(\prod e^{\alpha \gamma_i X_i})$ is the product of the expectations $\prod \mathbb{E}(e^{\alpha \gamma_i X_i})$.
>> [!info]- Source
>> [[obj_042 - Proof of Proposition 5]]

> [!question]- 152. How is uniqueness proven in the strictly convex case?
> By contradiction: if two distinct global minimizers existed, the function value at their midpoint would be strictly lower than the minimum, which is impossible.
>> [!info]- Source
>> [[obj_073 - Proof of Theorem 13]]

> [!question]- 153. How is the gradient limit used in this proof?
> By rearranging the convexity inequality and taking the limit as the step size $\lambda \to 0$, the difference quotient converges to the directional derivative, which is the inner product with the gradient.
>> [!info]- Source
>> [[obj_075 - Proof of Theorem 14]]

> [!question]- 154. How does the stationary condition help in the proof of Theorem 19?
> The stationary condition $\partial_w L(w, \xi^*, \nu^*) = 0$ implies that $w^*$ is a global minimizer of the convex function $w \to L(w, \xi^*, \nu^*)$.
>> [!info]- Source
>> [[obj_097 - Proof of Theorem 19]]

> [!question]- 155. What is the significance of showing that $f(w_k)$ is a strictly decreasing sequence in the proof?
> It ensures that every step of the algorithm makes progress towards the minimum, which is necessary for establishing the convergence of the sequence iterates.
>> [!info]- Source
>> [[obj_123 - Proof of Theorem 22]]

> [!question]- 156. How is the $O(1/k)$ rate established for smooth functions?
> By summing the progress made in each step $f(w_{k+1}) - f(w^*)$ and applying the 3-point inequality, we arrive at a bound $f(w_T) - f(w^*) \le (L/2T) \|w_0 - w^*\|^2$.
>> [!info]- Source
>> [[obj_123 - Proof of Theorem 22]]

> [!question]- 157. How is the randomness of the sampled index handled in the proof of Theorem 23?
> The proof uses conditional expectation $\mathbb{E}_k$ (conditioned on all previous iterates) to leverage the property that $\mathbb{E}_k(\nabla f_{i_k}(w_k)) = \nabla f(w_k)$.
>> [!info]- Source
>> [[obj_130 - Proof of Theorem 23]]

> [!question]- 158. What role does the assumption $\|\nabla f_i(w)\|^2 \le C^2$ play in this proof?
> It bounds the second moment of the stochastic gradient, ensuring the 'noise' term in the recursion does not grow uncontrollably.
>> [!info]- Source
>> [[obj_130 - Proof of Theorem 23]]

> [!question]- 159. Why is the KL divergence used in the proof of Theorem 25?
> Because KL divergence is non-negative and zero if and only if the distributions are equal, it provides a tool to establish the global lower bound of the objective.
>> [!info]- Source
>> [[obj_167 - Proof of Theorem 25]]

> [!question]- 160. Why is the union bound necessary in the proof of Theorem 7?
> Because the ERM $\hat{h}$ depends on the data, we must bound the risk for all possible hypotheses in the finite set $\mathcal{H}$ simultaneously.
>> [!info]- Source
>> [[obj_046 - Proof of Theorem 7]]

> [!question]- 161. State the rule for the Bayes classifier $h^*(x)$ for binary labels $\{0, 1\}$.
> $h^*(x) = 1$ if $\eta(x) > 1/2$, and $h^*(x) = 0$ if $\eta(x) \leq 1/2$, where $\eta(x) = \mathbb{P}(Y = 1 \mid X = x)$.
>> [!info]- Source
>> [[obj_012 - Proposition 1]]

> [!question]- 162. What is the mathematical bound provided by Proposition 11?
> $\mathbb{E}(\max_j W_j) \le \sigma \sqrt{2 \log d}$.
>> [!info]- Source
>> [[obj_060 - Proposition 11]]

> [!question]- 163. What is the necessary and sufficient condition for optimality in unconstrained convex optimization?
> $\nabla f(w^*) = 0$.
>> [!info]- Source
>> [[obj_080 - Proposition 15 (First order optimality condition - unconstrained)]]

> [!question]- 164. What is the condition for optimality in constrained convex optimization?
> For all $w \in \mathcal{F}$, we must have $\nabla f(w^*)^\top (w - w^*) \geq 0$.
>> [!info]- Source
>> [[obj_081 - Proposition 16 (First order optimality condition - constrained)]]

> [!question]- 165. What is the range of stepsize $\tau_k$ that guarantees convergence in Proposition 20?
> The stepsize must satisfy $0 < \tau_{min} \le \tau_k \le \tau_{max} < 2/L$, where $L$ is the largest eigenvalue of the matrix $C$.
>> [!info]- Source
>> [[obj_114 - Proposition 20]]

> [!question]- 166. What is the optimal contraction constant $\tilde{\rho}$ according to Proposition 20?
> The optimal rate is $\tilde{\rho} = (L - \mu)/(L + \mu)$, achieved when $\tau_k = 2/(L + \mu)$.
>> [!info]- Source
>> [[obj_114 - Proposition 20]]

> [!question]- 167. What is the lower bound provided by $\mu$-strong convexity in Proposition 21?
> The lower bound is $f(y) \ge f(x) + \langle \nabla f(x), y - x \rangle + \frac{\mu}{2}\|x - y\|^2$.
>> [!info]- Source
>> [[obj_117 - Proposition 21]]

> [!question]- 168. What is the upper bound provided by $L$-Lipschitz smoothness in Proposition 21?
> The upper bound is $f(y) \le f(x) + \langle \nabla f(x), y - x \rangle + \frac{L}{2}\|x - y\|^2$.
>> [!info]- Source
>> [[obj_117 - Proposition 21]]

> [!question]- 169. Under what condition does equality hold in the ELBO inequality?
> Equality holds when the approximate posterior $q$ equals the true posterior distribution $p(z|x)$.
>> [!info]- Source
>> [[obj_175 - Proposition 26 (ELBO)]]

> [!question]- 170. Given a sub-Gaussian variable $W$, what is the upper bound on $\mathbb{P}(|W - \mathbb{E}(W)| \ge t)$?
> $2 \exp(-t^2/(2\sigma^2))$.
>> [!info]- Source
>> [[obj_038 - Proposition 3]]

> [!question]- 171. What is the primary difference between a regression problem and a classification problem?
> In a regression problem, the output variable $Y$ is quantitative (numerical), whereas in a classification problem, the output variable is categorical (e.g., $\{ -1, 1 \}$).
>> [!info]- Source
>> [[obj_003 - Regression vs classification]]

> [!question]- 172. Provide an example of a regression task from the notes.
> Predicting the price of a house based on the number of bedrooms and other features is a regression task.
>> [!info]- Source
>> [[obj_003 - Regression vs classification]]

> [!question]- 173. How is Risk expressed when using the unit loss function?
> Under unit loss $\ell(z, y) = \mathbf{1}(z \neq y)$, the Risk is the misclassification probability: $R(h) = \mathbb{P}(h(X) \neq Y)$.
>> [!info]- Source
>> [[obj_007 - Remark 1]]

> [!question]- 174. How does Proposition 16 reduce to Proposition 15 in the unconstrained case?
> In the unconstrained case, we can choose $w - w^*$ to be any vector $v$. If $\nabla f(w^*)^ op v \geq 0$ for all $v$, then it must be that $\nabla f(w^*) = 0$.
>> [!info]- Source
>> [[obj_082 - Remark 10]]

> [!question]- 175. Is a kernel SVM a linear or nonlinear model?
> It is nonlinear in the input features but remains a linear model with respect to its parameters $w$ and $b$.
>> [!info]- Source
>> [[obj_103 - Remark 11]]

> [!question]- 176. What is the connection between SVMs and Neural Networks?
> An SVM with a sigmoid kernel can be viewed as a simple neural network where the hidden layer neurons use the training data points as their internal weights.
>> [!info]- Source
>> [[obj_107 - Remark 12]]

> [!question]- 177. How is the inverse condition number $\kappa$ defined in Remark 13?
> It is defined as the ratio of the smallest to largest eigenvalues of the Hessian matrix, $\kappa = \mu / L$.
>> [!info]- Source
>> [[obj_115 - Remark 13]]

> [!question]- 178. What happens to the convergence rate when the inverse condition number $\kappa$ is very small?
> When $\kappa$ is small, the contraction constant $\rho \approx 1 - 2\kappa$ becomes very close to 1, leading to very slow convergence.
>> [!info]- Source
>> [[obj_115 - Remark 13]]

> [!question]- 179. For a twice-differentiable function, how is $\mu$-strong convexity expressed in terms of the Hessian?
> It is equivalent to the condition $\mu I \preceq \nabla^2 f(x)$ for all $x$, meaning all eigenvalues of the Hessian are at least $\mu$.
>> [!info]- Source
>> [[obj_120 - Remark 14]]

> [!question]- 180. What is the Hessian-based equivalent for $L$-Lipschitz smoothness?
> It is equivalent to the condition $\nabla^2 f(x) \preceq L I$ for all $x$, meaning all eigenvalues of the Hessian are at most $L$.
>> [!info]- Source
>> [[obj_120 - Remark 14]]

> [!question]- 181. Why is SGD often preferred over GD in practice even if its theoretical convergence rate is slower?
> Because SGD iterations are much cheaper, allowing the algorithm to make significant progress toward the optimum long before GD has even finished a single full pass through a large dataset.
>> [!info]- Source
>> [[obj_131 - Remark 16]]

> [!question]- 182. What is the expected convergence rate for SGD on a smooth, convex (but not strongly convex) objective?
> The expected objective gap $\mathbb{E}(f(w_k)) - \min f(w)$ converges at a rate of $O(1/\sqrt{k})$.
>> [!info]- Source
>> [[obj_132 - Remark 17]]

> [!question]- 183. How can a ReLU network approximate a sigmoidal function for the purposes of the theorem?
> By defining $\sigma_1(r) = \text{ReLU}(z) - \text{ReLU}(z-1)$, which acts as a ramp function that is 0 for $r < 0$ and 1 for $r > 1$.
>> [!info]- Source
>> [[obj_139 - Remark 18]]

> [!question]- 184. What does each $\dot{z}_k$ represent if inputs are initialized with $\dot{z}_{-i} = r_i$?
> It represents the directional derivative $r^\top \nabla z_k$.
>> [!info]- Source
>> [[obj_146 - Remark 19]]

> [!question]- 185. What function minimizes the squared error risk in regression?
> The risk is minimized by the conditional expectation $h(x) = \mathbb{E}(Y \mid X = x)$.
>> [!info]- Source
>> [[obj_008 - Remark 2]]

> [!question]- 186. What is the cost of computing the full Jacobian using forward mode?
> $O(s)$, assuming the cost of evaluating the function once is $O(1)$ and $s$ is the number of inputs.
>> [!info]- Source
>> [[obj_147 - Remark 20]]

> [!question]- 187. How can a dual number $a + b\varepsilon$ be represented as a matrix?
> As the $2 \times 2$ matrix $\begin{pmatrix} a & 0 \\ b & a \end{pmatrix}$.
>> [!info]- Source
>> [[obj_148 - Remark 21]]

> [!question]- 188. What quantity is computed at the end of the backward pass in Remark 22?
> The vector $(J^\top r)_i$, where $J$ is the Jacobian of the function.
>> [!info]- Source
>> [[obj_153 - Remark 22]]

> [!question]- 189. Under what condition is reverse mode more efficient than forward mode?
> When the number of inputs $s$ is greater than the number of outputs $t$.
>> [!info]- Source
>> [[obj_154 - Remark 23 (Computational complexity)]]

> [!question]- 190. Why is the Bayes classifier called a 'maximum a posteriori estimator'?
> Because it chooses the label $y$ that maximizes the posterior probability $\mathbb{P}(Y = y \mid X = x)$.
>> [!info]- Source
>> [[obj_013 - Remark 3]]

> [!question]- 191. What is the integral expression for the Bayes risk $R(h^*)$?
> $R(h^*) = \mathbb{E}(\min(\eta(X), 1 - \eta(X)))$.
>> [!info]- Source
>> [[obj_016 - Remark 4]]

> [!question]- 192. What is the expectation of the empirical risk $\hat{R}(h)$ for a fixed $h$?
> $\mathbb{E}(\hat{R}(h)) = R(h)$, assuming the data are i.i.d.
>> [!info]- Source
>> [[obj_020 - Remark 5]]

> [!question]- 193. If $W$ is sub-Gaussian with parameter $\sigma$, is it also sub-Gaussian with parameter $\sigma' > \sigma$?
> Yes, the sub-Gaussian property is monotonic in the parameter.
>> [!info]- Source
>> [[obj_039 - Remark 6]]

> [!question]- 194. What is the main idea behind structural risk minimization?
> Choosing a hypothesis class $\mathcal{H}$ that minimizes the sum of the empirical risk and the estimation error bound.
>> [!info]- Source
>> [[obj_050 - Remark 7]]

> [!question]- 195. Is empirical Rademacher complexity dependent on the true probability distribution $P_0$?
> No, it depends only on the given dataset $z_{1:n}$.
>> [!info]- Source
>> [[obj_054 - Remark 8]]

> [!question]- 196. To show $VC(\mathcal{H}) \geq n$, do we need to show every set of size $n$ is shattered?
> No, we only need to find one set of $n$ distinct points that can be shattered.
>> [!info]- Source
>> [[obj_064 - Remark 9]]

> [!question]- 197. What is the formula for a skip connection in a ResNet?
> The connection takes the form $x_{k+1} = x_k + F_k(x_k)$, where $F_k$ is the residual learned by the layer.
>> [!info]- Source
>> [[obj_162 - Residual neural networks]]

> [!question]- 198. To what numerical method can ResNet skip connections be compared?
> Adding an artificial time step makes them similar to an explicit Euler discretization of an ordinary differential equation (ODE): $\dot{x}_t = F_t(x_t)$.
>> [!info]- Source
>> [[obj_162 - Residual neural networks]]

> [!question]- 199. Define the basic self-attention map $Atten_{Q,K,V}(X)(x)$.
> It is defined as $\sum_{i=1}^n \frac{\exp(x^\top Q^\top K x_i)}{\sum_{j=1}^n \exp(x^\top Q^\top K x_j)} V x_i$, where $Q, K, V$ are the query, key, and value matrices.
>> [!info]- Source
>> [[obj_164 - Self-attention]]

> [!question]- 200. What is Multi-head self-attention (MHSA)?
> MHSA is the combination of several self-attention maps, allowing the model to attend to information from different representation subspaces at different positions simultaneously.
>> [!info]- Source
>> [[obj_164 - Self-attention]]

> [!question]- 201. Who authored the primary book that these lecture notes largely follow?
> The notes largely follow the book by Martin Lotz.
>> [!info]- Source
>> [[obj_002 - Some excellent resources]]

> [!question]- 202. What coding packages are recommended for practical implementations in this course?
> Recommended packages include sklearn, pytorch, and JAX.
>> [!info]- Source
>> [[obj_002 - Some excellent resources]]

> [!question]- 203. What is the update rule for the SGD algorithm?
> The update rule is $w_{k+1} = w_k - \tau_k \nabla f_{i_k}(w_k)$, where $i_k$ is sampled uniformly at random from $\{1, ..., n\}$.
>> [!info]- Source
>> [[obj_126 - Stochastic gradient descent (SGD) Algorithm]]

> [!question]- 204. Why are the iterates $w_k$ in SGD considered random vectors?
> They are random because each update depends on the random index $i_k$ chosen at each iteration step $k$.
>> [!info]- Source
>> [[obj_126 - Stochastic gradient descent (SGD) Algorithm]]

> [!question]- 205. What is the general form of the Chernoff bound for a random variable $W$?
> $\mathbb{P}(W \ge t) \le \inf_{\alpha > 0} e^{-\alpha t} \mathbb{E}(e^{\alpha W})$.
>> [!info]- Source
>> [[obj_034 - The Chernoff bound]]

> [!question]- 206. What is the 'annoying thing' about maximizing the log-likelihood $p_\theta$ directly?
> Since $p_\theta$ is defined as an expectation $\mathbb{E}_z$, there is an expectation inside the log, which is difficult to optimize. Moving the log inside via Jensen's inequality creates a tractable bound.
>> [!info]- Source
>> [[obj_176 - The evidence lower bound (ELBO)]]

> [!question]- 207. What is the complementary slackness condition?
> It is the condition that $\nu^*_i f_i(w^*) = 0$, meaning that either the $i$-th constraint is active ($f_i(w^*) = 0$) or the corresponding multiplier is zero ($\nu^*_i = 0$).
>> [!info]- Source
>> [[obj_094 - The Karush-Kuhn-Tucker (KKT) conditions discussion]]

> [!question]- 208. What is the standard loss function used for regression in these notes?
> The standard regression loss is the squared error: $\ell(h(x), y) = (h(x) - y)^2$.
>> [!info]- Source
>> [[obj_004 - The loss function]]

> [!question]- 209. Define the unit loss function for classification.
> The unit loss is defined as $\ell(h(x), y) = 0$ if $h(x) = y$ and $1$ if $h(x) \neq y$.
>> [!info]- Source
>> [[obj_004 - The loss function]]

> [!question]- 210. State the formal integral definition of Risk $R(h)$.
> $$
> R(h) := \int_{\mathcal{X} \times \mathcal{Y}} \ell(h(x), y) dP_0(x, y) = \mathbb{E}(\ell(h(X), Y))
> $$
>> [!info]- Source
>> [[obj_009 - The Risk]]

> [!question]- 211. What condition on a convex function ensures its minimizer is unique?
> Strict convexity ensures uniqueness.
>> [!info]- Source
>> [[obj_074 - Theorem 13]]

> [!question]- 212. What is the second-order characterization of convexity?
> A twice-differentiable function is convex if and only if its Hessian $\nabla^2 f(v)$ is positive semi-definite for all $v$.
>> [!info]- Source
>> [[obj_076 - Theorem 14 (Characterization of convexity via differentiability)]]

> [!question]- 213. What are the requirements for Slater's condition?
> The objective and inequality constraints must be convex, and there must exist a point $w$ such that $Aw=b$ and $f_i(w) < 0$ for all inequality constraints.
>> [!info]- Source
>> [[obj_095 - Theorem 18 (Slater's constraint quantification)]]

> [!question]- 214. Are KKT conditions sufficient for optimality?
> Yes, if the optimization problem is convex and strong duality holds, then any point satisfying the KKT conditions is a primal-dual optimal solution.
>> [!info]- Source
>> [[obj_096 - Theorem 19 (Karush-Kuhn-Tucker)]]

> [!question]- 215. List the four main KKT conditions.
> 1. Stationary condition; 2. Primal feasibility; 3. Dual feasibility; 4. Complementary slackness.
>> [!info]- Source
>> [[obj_096 - Theorem 19 (Karush-Kuhn-Tucker)]]

> [!question]- 216. What is the philosophical implication of the No Free Lunch theorem?
> It implies that learning is impossible without assumptions; you cannot have a universal algorithm that learns optimally for every possible distribution.
>> [!info]- Source
>> [[obj_026 - Theorem 2]]

> [!question]- 217. What is the convergence rate of gradient descent if a function is only Lipschitz smooth?
> The objective value $f(w_k)$ converges to the minimum $f(w^*)$ at a rate of $O(1/k)$.
>> [!info]- Source
>> [[obj_121 - Theorem 22]]

> [!question]- 218. Under what conditions does gradient descent achieve linear convergence?
> Linear convergence is achieved when the function is both $L$-Lipschitz smooth and $\mu$-strongly convex.
>> [!info]- Source
>> [[obj_121 - Theorem 22]]

> [!question]- 219. What is the expected convergence rate of SGD for strongly convex functions according to Theorem 23?
> The rate is $\mathbb{E}(\|w_k - w^*\|^2) \le R/(k+1)$, which is a sublinear $O(1/k)$ rate.
>> [!info]- Source
>> [[obj_129 - Theorem 23]]

> [!question]- 220. What additional assumption about the gradients is required for Theorem 23?
> It is assumed that the norm of the gradients is bounded, i.e., $\|\nabla f_i(w)\|^2 \le C^2$ for all $i$.
>> [!info]- Source
>> [[obj_129 - Theorem 23]]

> [!question]- 221. What are the conditions on the activation function $\sigma$ in Theorem 24?
> It must be non-decreasing, continuous, and satisfy $\lim_{r \to \infty} \sigma(r) = 1$ and $\lim_{r \to -\infty} \sigma(r) = 0$.
>> [!info]- Source
>> [[obj_140 - Theorem 24 (The Universal Approximation theorem)]]

> [!question]- 222. Does Theorem 24 guarantee that we can find the required weights efficiently?
> No, it only guarantees their existence, not an efficient algorithm for finding them (though in practice we use SGD).
>> [!info]- Source
>> [[obj_140 - Theorem 24 (The Universal Approximation theorem)]]

> [!question]- 223. What is the optimal discriminator $D^*$ for a fixed generator $G$?
> $D^*(x) = \frac{\rho_X(x)}{\rho_X(x) + \rho_G(x)}$, where $\rho_G$ is the pushforward measure of the generator.
>> [!info]- Source
>> [[obj_166 - Theorem 25]]

> [!question]- 224. What is the exact dependence of the estimation error on the class size $|\mathcal{H}|$ in Theorem 7?
> The error depends on $\sqrt{\log(|\mathcal{H}|)}$.
>> [!info]- Source
>> [[obj_047 - Theorem 7]]

> [!question]- 225. How does the range $L$ of the loss function affect the error bound?
> The bound is directly proportional to $L$, meaning higher variance in loss values leads to looser guarantees.
>> [!info]- Source
>> [[obj_047 - Theorem 7]]

> [!question]- 226. What is the mathematical statement of Theorem 8?
> $\mathbb{E}(\sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n (\mathbb{E}f(Z_i) - f(Z_i))) \le 2\mathcal{R}_n(\mathcal{F})$.
>> [!info]- Source
>> [[obj_055 - Theorem 8]]

> [!question]- 227. How does Theorem 9 bound the estimation error?
> It shows that $\mathbb{E}(R(\hat{h}) - R(\bar{h})) \le 2\mathcal{R}_n(\mathcal{F})$, where $\mathcal{F}$ is the class of functions representing the loss of each hypothesis.
>> [!info]- Source
>> [[obj_056 - Theorem 9]]
