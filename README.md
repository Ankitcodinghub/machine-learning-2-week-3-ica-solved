# machine-learning-2-week-3-ica-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 2 Week 3-ICA Solved](https://www.ankitcodinghub.com/product/machine-learning-2-week-3-ica-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98825&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 2 Week 3-ICA Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column"></div>
<div class="column">
Exercise Sheet 3

</div>
</div>
<div class="layoutArea">
<div class="column">
Exercise 1: Maximum Entropy Distributions (5 + 10 + 5 P)

Consider a discrete random variable X defined over positive numbers N, each of them occuring with respec- tive probability Pr(X = k). The entropy H(X) is given by

H(X) = X1 Pr(X = k) logPr(X = k) k=0

We would like to find the probability distribution that maximizes the entropy H(X) under the expectation

</div>
</div>
<div class="layoutArea">
<div class="column">
constraint

</div>
<div class="column">
X1

E[X]= kPr(X =k)=1.

k=0

</div>
</div>
<div class="layoutArea">
<div class="column">
and forcing probabilities to be strictly positive, i.e. Pr(X = k) &gt; 0 for all k 2 N. The latter can be enforced using the reparameterization Pr(X = k) = exp(sk). To enforce that probability values sum to one, we can

impose the additional constraint

X1 Pr(X=k)=1.

k=0

The problem of finding the maximum-entropy distribution can therefore be modeled as a constrained opti-

mization problem, and such problems can be solved using the method of Lagrange multipliers.

(a) Write the Lagrange function L((sk)k2N,1,2) corresponding to the constrained optimization problem above, where 1,2 2 R are used to incorporate the sum-to-one and the expectation constraints respec- tively.

(b) Show that the probability distribution that maximizes the objective H(X) has the form: Pr(X = k) = ↵ · k

(c) Show that ↵ = 0.5 and = 0.5, i.e. the probability distribution is geometric with failure and success probabilities 0.5, and k denoting the number of failures before success. (Hint: you can make use of the geometric series and Gabriel’s staircase series).

Exercise 2: Independent Components in Two Dimensions (5 + 10 + 10 P)

High entropy can be seen as the result of superposing many independent low-entropy sources. Independent component analysis (ICA) aims to recover the independent sources from the data by finding projections of the data that have low entropy, i.e. that diverge the most from a Gaussian probability distribution.

In the following, we consider a simple two-dimensional problem where we are given a joint probability distribution p(x, y) = p(x) p(y|x) with

p(x) ⇠ N (0, 1),

p ( y | x ) = 12 ( y x ) + 12 ( y + x ) ,

where (·) denotes the Dirac delta function. A useful property of linear component analysis for two- dimensional probability distributions is that the set of all possible directions to look for in R2 is directly

</div>
</div>
<div class="layoutArea">
<div class="column">
given by

</div>
<div class="column">
⇢✓cos✓◆, 0  ✓ &lt; 2⇡. sin ✓

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
The projection of the random vector (x,y) on a particular component can therefore be expressed as a function of ✓:

z(✓) = x cos(✓) + y sin(✓).

As a result, ICA in the two-dimensional space is reduced to finding the values of the parameter ✓ 2 [0, 2⇡[

that maximize a certain objective J(z(✓)).

<ol>
<li>(a) &nbsp;Sketch the joint probability distribution p(x, y), along with the projections z(✓) of this distribution for angles
✓ = 0, ✓ = ⇡/8 and ✓ = ⇡/4.
</li>
<li>(b) &nbsp;Find the principal components of p(x, y). That is, find the values of the parameter ✓ 2 [0, 2⇡[ that maximize
the variance of the projected data z(✓).
</li>
<li>(c) &nbsp;Find the independent components of p(x, y), more specifically, find the values of the parameter ✓ 2 [0, 2⇡[ that maximize the non-Gaussianity of z(✓). We use as a measure of non-Gaussianity the excess kurtosis</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
defined as

</div>
<div class="column">
kurt [z(✓)] = E⇥(z(✓) E[z(✓)])4⇤ 3. (Var[z(✓)])2

</div>
</div>
<div class="layoutArea">
<div class="column">
Exercise 3: Deriving a Special Case of FastICA (5 + 5 + 5 + 10 P)

We consider our input data x to be in Rd and coming from some distribution p(x). We assume we have applied as an initial step the centering and whitening procedures so that:

E[x] = 0 and E[xx&gt;] = I.

To extract an independent component, we would like to find a unit vector w (i.e. kwk2 = 1) such that the

excess kurtosis of the projected data:

kurt [w&gt;x] = E⇥(w&gt;x E[w&gt;x])4⇤ 3.

</div>
</div>
<div class="layoutArea">
<div class="column">
(Var[w&gt;x])2

<ol>
<li>(a) &nbsp;Show that for any w subject to kwk2 = 1, the projection z = w&gt;x has mean 0 and variance 1.</li>
<li>(b) &nbsp;Show using the method of Lagrange multipliers that the projection w that maximizes kurt [w&gt;x] is a solution of the equation
w = E[x · (w&gt;x)3] where can be resolved by enforcing the constraint kwk2 = 1.
</li>
<li>(c) &nbsp;The solution of the previous equation cannot be found analytically, and must instead be solved iteratively using e.g. the Newton’s method. The Newton’s method assumes that the equation is given in the form F(w) = 0 and then defines the iteration as w+ = w J(w)1F(w) where w+ denotes the next iterate and where J is the Jacobian associated to the function F. Show that the Newton iteration in our case takes the form:
w+ = w (E[3xx&gt;(w&gt;x)2 I])1 · (E[x · (w&gt;x)3] w)
</li>
<li>(d) &nbsp;Show that when making the decorrelation approximation E[xx&gt;(w&gt;x)2] = E[xx&gt;] · E[(w&gt;x)2], the Newton</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
is maximized.

</div>
</div>
<div class="layoutArea">
<div class="column">
iteration further reduces to

where is some constant factor. (The constant factor does not need to be resolved since we subsequently

</div>
</div>
<div class="layoutArea">
<div class="column">
w+ =·(E[x·(w&gt;x)3]3w) apply the normalization step w+ w+/kw+k.)

</div>
</div>
<div class="layoutArea">
<div class="column">
Exercise 4: Programming (30 P)

Download the programming files on ISIS and follow the instructions.

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="layoutArea">
<div class="column">
Exercise sheet 3 (programming) [SoSe 2021] Machine Learning 2

</div>
</div>
<div class="layoutArea">
<div class="column">
Independent Component Analysis

In this exercise, you will implement an ICA algorithm similar to the FastICA method described in the paper “A. Hyvärinen and E. Oja. 2000. Independent component analysis: algorithms and applications” linked from ISIS, and apply it to model the independent components of a distribution of image patches.

In [1]:

import numpy

import matplotlib

%matplotlib inline

from matplotlib import pyplot as plt import sklearn

import sklearn.datasets

import sklearn.feature_extraction.image import utils

As a first step, we take a sample image, extract a collection of $(8 \times 8)$ patches from it and plot them.

In [2]:

<pre>I = sklearn.datasets.load_sample_image('china.jpg')
X = sklearn.feature_extraction.image.extract_patches_2d(I, (8,8), max_patches=10000, random_state=0)
utils.showimage(I)
utils.showpatches(X)
</pre>
As a starting point, the patches we have extracted are flattened to appear as abstract input vectors of $8 \times 8 \times 3 = 192$ dimensions. The input data is then centered and standardized.

In [3]:

<pre>X = X.reshape(len(X),-1)
X = X - X.mean(axis=0)
X = X / X.std()
</pre>
</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="section">
<div class="layoutArea">
<div class="column">
Whitening (10 P)

A precondition for applying the ICA procedure is that the input data has variance $1$ under any projection. This can be achieved by whitening, which is a transformation $\mathcal{W}:\mathbb{R}^d \to \mathbb{R}^d$ with $z = \mathcal{W}(x)$ such that $\mathrm{E}[zz^\top] = I$.

A simple procedure for whitening a collection of data points $x_1,\dots,x_N$ (assumed to be centered) first computes the PCA components $u_1,\dots,u_d$ of the data and then applies the following three consecutive steps:

1. project the data on the PCA components i.e. $p_{n,i} = x_n^\top u_i$.

2. divide the projected data by the standard deviation in PCA space, i.e. $\tilde{p}_{n,i} = p_{n,i} / \text{std}(p_{:,i})$ 3. backproject to the input space $z_n = \sum_i \tilde{p}_{n,i} u_i$.

Task:

Implement this whitening procedure, in particular, write a function that receives the input data matrix and returns the matrix containing all whitened data points.

For efficiency, your whitening procedure should be implemented in matrix form.

In [4]:

<pre>##### REPLACE BY YOUR CODE
</pre>
<pre>import solutions
</pre>
<pre>Z = solutions.whitening(X)
</pre>
#####

The code below verifies graphically that whitening has removed correlations between the different input dimensions:

In [5]:

<pre>f = plt.figure(figsize=(8,4))
p = f.add_subplot(1,2,1); p.set_title('before whitening')
p.imshow(numpy.dot(X.T,X)/len(X))
p = f.add_subplot(1,2,2); p.set_title('after whitening')
p.imshow(numpy.dot(Z.T,Z)/len(Z))
plt.show()
</pre>
Finally, to get visual picture of what will enter into our ICA algorithm, the whitened data can be visualized in the same way as the original input data.

In [6]:

<pre>utils.showpatches(X)
utils.showpatches(Z)
</pre>
</div>
</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="section">
<div class="layoutArea">
<div class="column">
We observe that all high constrasts and spatial correlations have been removed after whitening. Remaining patterns include high-frequency textures and oriented edges of different colors.

Implementing ICA (20 P)

We now would like to learn $h=64$ independent components of the distribution of whitened image patches. For this, we adopt a procedure similar to the FastICA procedure described in the paper above. In particular, we start with random weights $w_1,\dots,w_h \in \mathbb{R}^d$ and iterate multiple times the sequence of operations:

1. $\forall_{i=1}^d~w_i = \mathbb{E}[x \cdot g(w_i^\top x)] – w_i \cdot \mathbb{E}[ g'(w_i^\top x)]$ 2. $w_1,\dots,w_h=\text{decorrelate}\{w_1,\dots,w_h\}$

where $\mathbb{E}[\cdot]$ denotes the expectation with the data distribution.

The first step increases non-Gaussianity of the projected data. Here, we will make use of the nonquadratic function $G(x) = \frac1a \log \cosh (a x)$ with $a=1.5$. This function admits as a derivative the function $g(x) = \tanh(a x)$, and as a double derivative the function $g'(x) = a \cdot (1-\tanh^2(a x))$.

The second step enforces that the learned projections are decorrelated, i.e.\ $w_i^\top w_j = 1_{i=j}$. It will be implemented by calling in an appropriate manner the whitening procedure which we have already implemented to decorrelate the different input dimensions.

This procedure minimizes the non-Gaussianity of the projected data as measured by the objective:

$$ J(w) = \sum_{i=1}^h (\mathbb{E}[G(w_i^\top x)] – \mathbb{E}[G(\varepsilon)])^2 \qquad \text{where} \quad \varepsilon \sim \mathcal{N}(0,1). $$

Task:

Implement the ICA procedure described above, run it for 200 iterations, and print the value of the objective function every 25 iterations.

In order to keep the learning procedure computationally affordable, the code must be parallelized, in particular, make use of numpy matrix multiplications instead of loops whenever it is possible.

In [7]:

<pre>##### REPLACE BY YOUR CODE
</pre>
<pre>import solutions
</pre>
<pre>W = solutions.ICA(Z)
</pre>
#####

it=25 J(W) = 1.47 it=50 J(W) = 1.82 it=75 J(W) = 1.96 it=100 J(W) = 2.03 it=125 J(W) = 2.07 it=150 J(W) = 2.09 it=175 J(W) = 2.10 it=200 J(W) = 2.12

Because the learned ICA components are in a space of same dimensions as the input data, they can also be visualized as image patches.

<pre>In [8]:
utils.showpatches(W)
</pre>
We observe that an interesting decomposition appears, composed of frequency filters, edges filters and localized texture filters. The decomposition further aligns on specific directions of the RGB space, specifically yellow/blue and red/cyan.

To verify that strongly non-Gaussian components have been learned, we build a histogram of projections on the various ICA components and compare it to histograms for random projections.

</div>
</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="section">
<div class="layoutArea">
<div class="column">
In [9]:

import numpy

plt.figure(figsize=(7,2)) for i in range(64):

plt.hist(numpy.dot(Z,W[i]),bins=numpy.linspace(-12.5,12.5,26),histtype=’step’,log=True) plt.title(‘ICA projections’)

plt.show()

plt.figure(figsize=(7,2)) for i in range(64):

<pre>    R = numpy.random.mtrand.RandomState(i).normal(0,1,Z.shape[1])
</pre>
plt.hist(numpy.dot(Z,R/(R**2).sum()**.5),bins=numpy.linspace(-12.5,12.5,26),histtype=’step’,log=True) plt.title(‘random projections’)

plt.show()

We observe that the ICA projections have much heavier tails. This is a typical characteristic of independent components of a data distribution.

</div>
</div>
</div>
</div>
<div class="page" title="Page 7"></div>
<div class="page" title="Page 8"></div>
<div class="page" title="Page 9"></div>
<div class="page" title="Page 10"></div>
<div class="page" title="Page 11"></div>
