# Formulas To Add

Several key formulas from the papers would significantly enhance your research prospect by providing mathematical rigor and demonstrating your technical understanding of the analytical methods. Here are the most important formulas to include:

## **1\. From Tanaka-Ishii & Bunde (2016) \- Long-range memory:**

**Exceedance Probability Formula (Weibull function):**

   ```formula
   SQ(r) = exp(-b(β)(r/RQ)^β)
   ```

Where SQ(r) is the probability that the length of an interval exceeds r, β is the exponent (found to be around 0.7-0.8), and RQ is the mean interval length.

**Autocorrelation Function for Long-range Memory:**

   ```formula
   CQ(s) = CQ(1)s^(-γ)
   ```

Where γ is between 0.14 and 0.48 for literary texts, s is the distance, and CQ(s) measures the correlation between intervals separated by distance s.

**White Noise Fraction Estimation:**

   ```formula
   a = 1 / (1 + √[CQ(1)/(CQ^(0)(1) - CQ(1))])
   ```

Where a represents the fraction of white noise in the intervals (found to be 0.55-0.69 for natural language texts).

### **2\. From Tanaka-Ishii & Kobayashi (2018) \- Taylor's law:**

**Taylor's Law Basic Relationship:**

   ```formula
   σ ∝ μ^α
   ```

Where μ is the mean frequency of word occurrence, σ is the standard deviation, and α is the Taylor exponent (approximately 0.58 for written texts).

**Taylor Exponent Calculation:**

   ```formula
   α = log(σk/σ1) / log(μk/μ1)
   ```

Where σk and μk are the standard deviation and mean for a specific word k.

### **3\. From Takahira, Tanaka-Ishii & Dębowski (2016) \- Entropy rate:**

**Stretched Exponential Extrapolation Function:**

   ```formula
   f3(n) = exp(An^(β-1) + h')
   ```

Where n is the text length, β is around 0.7, and the entropy rate h \= exp(h').

**Entropy Rate Definition:**

   ```formula
   h = lim(n→∞) H(X_1^n) / n
   ```

Where H(X\_1^n) is the Shannon entropy of a sequence of length n.

### **4\. From Tanaka-Ishii (2021) \- Zipf's and Heaps' laws:**

**Zipf's Law:**

   ```formula
   f(r) ∝ r^(-η)
   ```

Where f(r) is the frequency of the word with rank r, and η is approximately 1.0.

**Heaps' Law:**

   ```formula
   v ∝ m^ξ
   ```

Where v is the vocabulary size and m is the text length, with ξ being the Heaps' exponent.

Including these formulas in your research prospect will:

1. Show your technical grasp of the analytical methods  
2. Allow you to precisely explain how you'll quantify changes across time periods  
3. Provide a clear framework for interpreting your pilot study results  
4. Demonstrate the mathematical rigor of your proposed approach

I would recommend placing these formulas in the Theoretical Foundation and Analytical Methods sections, with brief explanations of how each will be used to track changes over historical periods.

Including key formulas from the papers would strengthen your research prospect by demonstrating technical rigor and providing clear metrics for your diachronic analysis. Here are the most relevant formulas that would enhance your proposal:

1. **Taylor's Law Formula**:

   ```formula
   σ ∝ μᶜ
   ```

2. This power-law relationship between standard deviation (σ) and mean frequency (μ) with Taylor exponent ζ is fundamental to your analysis and appears to be one of the most effective metrics based on the papers.

3. **Autocorrelation Function for Long-Range Correlation**:

   ```formula
   c(s) = E[(xₜ - μ)(xₜ₊ₛ - μ)]/σ²
   ```

4. ```formula
   c(s) ∝ s⁻ᵝ, s > 0, 0 < ξ < 1
   ```

5. These formulas define the autocorrelation function and its expected power-law decay for long-range correlated systems.

6. **Ebeling's Method Formula**:

   ```formula
   m(l) = ∑ m₂(c, l) ∝ lⁿ
   ```

7. Where m₂(c, l) is the variance of counts y(c, l):

   ```formula
   m₂(c, l) = <y²(c, l)> - (<y(c, l)>)²
   ```

8. This shows the power-law relation between subsection length and variance in character occurrences.

9. **Zipf's Law and Heaps' Law**:

   ```formula
   f(r) ∝ r⁻ᵅ       (Zipf's law)
   v(n) ∝ nᵝ        (Heaps' law)
   ```

10. These classic formulas are essential for analyzing vocabulary distribution.

11. **Error Calculation Method**:

   ```formula
   ε(κ, c) ≡ √(∑ᵢ(log yᵢ - log cx_i^κ)²/N)
   ```

12. This formula from the Takahashi and Tanaka-Ishii paper shows how error is calculated for power-law exponents, which would be crucial for your quantitative analysis.

13. **White Noise Fraction Calculation**: The methodology for calculating the white noise fraction in texts would be valuable to include, though the full formula is complex and involves detrended fluctuation analysis.

14. **Strahler Number Relationship with Sentence Length**:

   ```formula
   S ≈ a log₂(L) + b
   ```

15. Where S is the Strahler number and L is sentence length, with a and b as constants. This logarithmic relationship is key to your analysis of cognitive constraints.

Including these formulas in appropriate sections of your research prospect would help clarify your methodological approach and give readers a concrete understanding of the quantitative measures you'll be using to track language change over time. You could place them within the Analytical Methods section, with brief explanations of how each will be applied to historical texts from different periods.

Adding key formulas from the Strahler number paper would strengthen your research prospect by demonstrating technical understanding and providing clear methodological details. Here are the most relevant formulas you could incorporate:

1. **Strahler Number Calculation Formula**:

   * For a leaf node v: S(v) \= 1  
   * For an inner node v with two children ch₁(v) and ch₂(v):  
     * If S(ch₁(v)) \== S(ch₂(v)), then S(v) \= S(ch₁(v)) \+ 1  
     * Otherwise, S(v) \= max(S(ch₁(v)), S(ch₂(v)))  
2. **The "it function"** used for calculating upper/lower limits of Strahler numbers:

   ```formula
   it(x, y) = {
   x + 1 if x == y
   max(x, y) otherwise
   }
   ```

3. **Logarithmic Growth Relationship** for Strahler numbers of random binary trees:

   ```formula
   E[R₂(n)] = log₄ n + 1 + o(1)
   ```

4. Where E\[R₂(n)\] is the mean Strahler number of binary trees with n leaves.

5. **Upper limit formula** for Strahler number of binary trees with n leaves:

   ```formula
   Upper limit = ⌊log₂ n⌋ + 1
   ```

6. **Dynamic Programming Calculation** for average upper/lower limits:

   ```formula
   Rₘₐₓ(n) = (∑ₚₘₐₓ pₘₐₓRₙ,ₚₘₐₓ)/(∑ₚₘₐₓ Rₙ,ₚₘₐₓ)
   Rₘᵢₙ(n) = (∑ₚₘᵢₙ pₘᵢₙRₙ,ₚₘᵢₙ)/(∑ₚₘᵢₙ Rₙ,ₚₘᵢₙ)
   ```

7. Where Rₙ,ₚₘₐₓ and Rₙ,ₚₘᵢₙ are the total number of trees of size n with upper limit pₘₐₓ and lower limit pₘᵢₙ, respectively.

Including these formulas would:

1. Make your methodology section more precise and implementable  
2. Demonstrate your technical understanding of the mathematical concepts  
3. Provide clear parameters that can be tracked across different time periods  
4. Show how you plan to connect sentence complexity to cognitive constraints quantitatively

These formulas could be presented in a dedicated "Mathematical Framework" subsection within your methodology, with explanations of how each will be applied to analyze historical texts.