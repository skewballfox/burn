use burn_backend::{Backend, tensor::Trigonometric};

use crate::Tensor;

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: Trigonometric<B>,
{
    /// Applies element wise cosine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \cos\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = cos(x_i)`")]
    pub fn cos(self) -> Self {
        Tensor::new(K::cos(self.primitive))
    }

    /// Applies element wise sine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \sin\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = sin(x_i)`")]
    pub fn sin(self) -> Self {
        Tensor::new(K::sin(self.primitive))
    }

    /// Applies element wise tangent operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \tan\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = tan(x_i)`")]
    pub fn tan(self) -> Self {
        Tensor::new(K::tan(self.primitive))
    }

    /// Applies element wise hyperbolic cosine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \cosh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = cosh(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 2.0], &device);
    ///     println!("{}", tensor.cosh()); // [1.0, 1.5430, 3.7621]
    /// }
    /// ```
    pub fn cosh(self) -> Self {
        Tensor::new(K::cosh(self.primitive))
    }

    /// Applies element wise hyperbolic sine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \sinh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = sinh(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 2.0], &device);
    ///     println!("{}", tensor.sinh()); // [0.0, -1.1752, 3.6269]
    /// }
    /// ```
    pub fn sinh(self) -> Self {
        Tensor::new(K::sinh(self.primitive))
    }

    /// Applies element wise hyperbolic tangent operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \tanh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = tanh(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 2.0], &device);
    ///     println!("{}", tensor.tanh()); // [0.0, -0.7616, 0.9640]
    /// }
    /// ```
    pub fn tanh(self) -> Self {
        Tensor::new(K::tanh(self.primitive))
    }

    /// Applies element wise inverse cosine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \acos\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = acos(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 1.0], &device);
    ///     println!("{}", tensor.acos()); // [1.5708, 3.1416, 0.0]
    /// }
    /// ```
    pub fn acos(self) -> Self {
        Tensor::new(K::acos(self.primitive))
    }

    /// Applies element wise inverse hyperbolic cosine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \acosh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = acosh(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([1.0, 2.0, 3.0], &device);
    ///     println!("{}", tensor.acosh()); // [0.0000, 1.3170, 1.7627]
    /// }
    /// ```
    pub fn acosh(self) -> Self {
        Tensor::new(K::acosh(self.primitive))
    }

    /// Applies element wise inverse sine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \asin\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = asin(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 1.0], &device);
    ///     println!("{}", tensor.asin()); // [ 0.0000, -1.5708,  1.5708]
    /// }
    /// ```
    pub fn asin(self) -> Self {
        Tensor::new(K::asin(self.primitive))
    }

    /// Applies element wise inverse hyperbolic sine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \asinh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = asinh(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 1.0], &device);
    ///     println!("{}", tensor.asinh()); // [ 0.0000, -0.8814,  0.8814]
    /// }
    /// ```
    pub fn asinh(self) -> Self {
        Tensor::new(K::asinh(self.primitive))
    }

    /// Applies element wise inverse tangent operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \atan\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = atan(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 2.0], &device);
    ///     println!("{}", tensor.atan()); // [ 0.0, -0.7854,  1.1071]
    /// }
    /// ```
    pub fn atan(self) -> Self {
        Tensor::new(K::atan(self.primitive))
    }

    /// Applies element wise inverse hyperbolic tangent operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \atanh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = atanh(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -0.5, 0.5], &device);
    ///     println!("{}", tensor.atanh()); // [ 0.0, -0.5493,  0.5493]
    /// }
    /// ```
    pub fn atanh(self) -> Self {
        Tensor::new(K::atanh(self.primitive))
    }

    /// Applies element wise inverse tangent operation using the signs of arguments to determine the correct quadrant.
    ///
    #[cfg_attr(doc, doc = r#"$z_i = \atan2\(y_i, x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`z_i = atan2(y_i, x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let lhs = Tensor::<B, 1>::from_data([-2.0, 2.0, -2.0], &device);
    ///     let rhs = Tensor::<B, 1>::from_data([1.0, -1.0, -1.0], &device);
    ///     println!("{}", lhs.atan2(rhs)); // [-1.1071,  2.0344, -2.0344]
    /// }
    /// ```
    pub fn atan2(self, other: Self) -> Self {
        Tensor::new(K::atan2(self.primitive, other.primitive))
    }
}
