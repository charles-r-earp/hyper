use std::ops::{Add, Sub};
use ndarray::{ArrayBase, Array2, Ix2, Data, azip};
use ndarray::linalg::Dot;
use rand::distributions::{Distribution, Normal};
use num_traits::{Zero, Float};
use num_traits::cast::{NumCast, ToPrimitive};

trait Build<X> {
  type P;
  fn build(&self, x: X) -> Self::P;
}

trait Forward<X, P> {
  type Y;
  fn forward(&self, x: X, p: P) -> Self::Y;
}

trait Backward<X, P, DY> {
  type DX;
  type DP;
  fn backward(&self, x: X, p: P, dy: DY) -> (Self::DX, Self::DP);
}

struct Norm;

struct Weight {
  c: usize
}

impl<A, X> Build<ArrayBase<X, Ix2>> for Weight
  where A: Float,
        X: Data<Elem=A> {
  type P = Array2<A>;
  fn build(&self, x: ArrayBase<X, Ix2>) -> Self::P {
    let ic = x.shape()[1];
    // he initialization
    let n = Normal::new(0., (2./(ic + self.c).to_f64().unwrap()).sqrt());
    let randn_fn = || A::from(n.sample(&mut rand::thread_rng())).unwrap();
    Self::P::from_shape_fn([ic, self.c], |_| randn_fn())
  }
}

impl<A, X, P> Forward<ArrayBase<X, Ix2>, ArrayBase<P, Ix2>> for Weight
  where A: Float + 'static,
        X: Data<Elem=A>,
        P: Data<Elem=A> {
  type Y = Array2<A>;
  fn forward(&self, x: ArrayBase<X, Ix2>, p: ArrayBase<P, Ix2>) -> Self::Y {
    x.dot(&p)
  }
} 

impl<A, X, P, DY> Backward<ArrayBase<X, Ix2>, ArrayBase<P, Ix2>, ArrayBase<DY, Ix2>> for Weight
  where A: Float + 'static,
        X: Data<Elem=A>,
        P: Data<Elem=A>,
        DY: Data<Elem=A> {
  type DX = Array2<A>;
  type DP = Array2<A>;
  fn backward(&self, x: ArrayBase<X, Ix2>, p: ArrayBase<P, Ix2>, dy: ArrayBase<DY, Ix2>) -> (Self::DX, Self::DP) {
    let dx = dy.dot(&p.t());
    let dp = x.t().dot(&dy);
    (dx, dp)
  }
}

struct Bias;

impl<A, X> Build<ArrayBase<X, Ix2>> for Bias
  where A: Float,
        X: Data<Elem=A> {
  type P = Array2<A>;
  fn build(&self, x: ArrayBase<X, Ix2>) -> Self::P {
    Self::P::zeros([1, x.shape()[1]])
  }
}

impl<A, X, P> Forward<ArrayBase<X, Ix2>, ArrayBase<P, Ix2>> for Bias
  where ArrayBase<X, Ix2>: Add<ArrayBase<P, Ix2>, Output=Array2<A>>,
        X: Data<Elem=A>,
        P: Data<Elem=A>,
        A: Float {
  type Y = Array2<A>;
  fn forward(&self, x: ArrayBase<X, Ix2>, p: ArrayBase<P, Ix2>) -> Self::Y {
    x + p
  }
}

impl<A, X, P, DY> Backward<ArrayBase<X, Ix2>, ArrayBase<P, Ix2>, ArrayBase<DY, Ix2>> for Bias
  where A: Float + 'static,
        X: Data<Elem=A>,
        P: Data<Elem=A>,
        DY: Data<Elem=A> {
  type DX = ArrayBase<DY, Ix2>;
  type DP = Array2<A>;
  fn backward(&self, x: ArrayBase<X, Ix2>, p: ArrayBase<P, Ix2>, dy: ArrayBase<DY, Ix2>) -> (Self::DX, Self::DP) {
    let dp = dy.t().dot(&Self::DP::ones(x.raw_dim()));
    let dx = dy;
    (dx, dp)
  }
}

struct L1Loss; 

impl<A, X, P> Forward<ArrayBase<X, Ix2>, ArrayBase<P, Ix2>> for L1Loss 
  where A: Float + Sub<A, Output=A>,
        X: Data<Elem=A>,
        P: Data<Elem=A> {
  type Y = Array2<A>;
  fn forward(&self, x: ArrayBase<X, Ix2>, p: ArrayBase<P, Ix2>) -> Self::Y {
    let mut y = unsafe { Self::Y::uninitialized(x.raw_dim()) };
    azip!(mut y, x, p in { *y = (x - p).abs() });
    y
  }
}

impl<A, X, P, DY> Backward<ArrayBase<X, Ix2>, ArrayBase<P, Ix2>, ArrayBase<DY, Ix2>> for L1Loss
  where A: Float,
        X: Data<Elem=A>,
        P: Data<Elem=A>,
        DY: Data<Elem=A> {
  type DX = Array2<A>;
  type DP = ();
  fn backward(&self, x: ArrayBase<X, Ix2>, p: ArrayBase<P, Ix2>, dy: ArrayBase<DY, Ix2>) -> (Self::DX, Self::DP) {
    let mut dx = unsafe { Self::DX::uninitialized(x.raw_dim()) };
    azip!(mut dx, x, p, dy in { 
      *dx = if x >= p { A::one() } else { -A::one() } 
    });
    (dx, ())
  }
}

struct Perceptron {
  w: Array2<f32>,
  b: Array2<f32>
}

impl Perceptron {
  fn new() -> Self {
    let fc = Weight{c: 1};
    let x = Array2::<f32>::ones([1, 2]);
    let w = fc.build(x.view());
    let x1 = fc.forward(x, w.view());
    let b = Bias.build(x1);
    Self{w, b}
  }
  fn pred(&self, x: Array2<f32>) -> Array2<f32> {
    let x1 = Weight{c: 1}.forward(x, self.w.view());
    Bias.forward(x1, self.b.view())
  } 
  fn eval(&self, x: Array2<f32>, t: Vec<usize>) -> (usize, f32) {
    let y = self.pred(x);
    let mut c = 0;
    let mut loss = 0.;
    y.iter().zip(t.iter())
            .for_each(|(y, t)| {
      if t == &1 {
        if y > &0.5 { c += 1; }
        loss += 1. - y;
      }
      else {
        if y <= &0.5 { c += 1; }
        loss += y;
      }
    });
    (c, loss)
  }         
}

struct Problem {
  m: Array2::<f32>,
  b: Array2::<f32>
}

fn main() {
  let fc = Weight{c: 1};
  let x = Array2::<f32>::ones([10, 2]);
  let w = fc.build(x.view());
  let x1 = fc.forward(x.view(), w.view());
  let b = Bias.build(x1.view());
  let x2 = Bias.forward(x1.clone(), b.view());
  let t = Array2::<f32>::ones([10, 1]);
  let x3 = L1Loss.forward(x2.view(), t.view()); 
  let dx3 = Array2::<f32>::ones(x3.raw_dim());
  let (dx2, _) = L1Loss.backward(x3.view(), t.view(), dx3.view());
  let (dx1, db) = Bias.backward(x1.view(), b.view(), dx2.view());
  println!("dx1:\n{:?}\nb:\n{:?}\ndb:\n{:?}", &dx1, &b, &db);
  let (dx0, dw) = fc.backward(x.view(), w.view(), dx1.view());
  println!("dx2:\n{:?}\nw:\n{:?}\ndw:\n{:?}", &dx0, &w, &dw);
}
