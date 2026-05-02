mod common;

use burn_tensor::{Complex, Int, Tensor, TensorData};
use common::*;

#[test]
fn test_complex_zeros() {
    let tensor = TestTensor::<2>::zeros([2, 3], &Default::default());
    let data = tensor.into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ],
        [
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_ones() {
    let tensor = TestTensor::<2>::ones([2, 2], &Default::default());
    let data = tensor.into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
        ],
        [
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_from_data() {
    let data = TensorData::from([[
        Complex::<f32> {
            real: 1.0,
            imag: 2.0,
        },
        Complex::<f32> {
            real: 3.0,
            imag: 4.0,
        },
    ]]);

    let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());
    let result = tensor.into_data();

    result.assert_eq(&data, false);
}

#[test]
fn test_complex_reshape() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
            Complex::<f32> {
                real: 5.0,
                imag: 6.0,
            },
            Complex::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ]]),
        &Default::default(),
    );

    let result: TestTensor<3> = tensor.reshape([2, 2, 1]);
    let data = result.into_data();

    let expected = TensorData::from([
        [
            [Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            }],
            [Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            }],
        ],
        [
            [Complex::<f32> {
                real: 5.0,
                imag: 6.0,
            }],
            [Complex::<f32> {
                real: 7.0,
                imag: 8.0,
            }],
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_transpose() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                Complex::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                Complex::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                Complex::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let result = tensor.transpose();
    let data = result.into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 5.0,
                imag: 6.0,
            },
        ],
        [
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
            Complex::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_swap_dims() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                Complex::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                Complex::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                Complex::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let data = tensor.swap_dims(0, 1).into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 5.0,
                imag: 6.0,
            },
        ],
        [
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
            Complex::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_slice() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                Complex::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                Complex::<f32> {
                    real: 2.0,
                    imag: 2.0,
                },
                Complex::<f32> {
                    real: 3.0,
                    imag: 3.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 4.0,
                    imag: 4.0,
                },
                Complex::<f32> {
                    real: 5.0,
                    imag: 5.0,
                },
                Complex::<f32> {
                    real: 6.0,
                    imag: 6.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let data = tensor.slice([0..1, 1..3]).into_data();

    let expected = TensorData::from([[
        Complex::<f32> {
            real: 2.0,
            imag: 2.0,
        },
        Complex::<f32> {
            real: 3.0,
            imag: 3.0,
        },
    ]]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_repeat_dim() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let data = tensor.repeat_dim(0, 2).into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
        [
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_cat() {
    let t1 = TestTensor::<2>::from_data(
        TensorData::from([[
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );
    let t2 = TestTensor::<2>::from_data(
        TensorData::from([[
            Complex::<f32> {
                real: 5.0,
                imag: 6.0,
            },
            Complex::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ]]),
        &Default::default(),
    );

    let data = TestTensor::<2>::cat(vec![t1, t2], 0).into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
        [
            Complex::<f32> {
                real: 5.0,
                imag: 6.0,
            },
            Complex::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_equal() {
    let t1 = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]),
        &Default::default(),
    );
    let t2 = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );

    t1.equal(t2)
        .into_data()
        .assert_eq(&TensorData::from([true, false]), false);
}

#[test]
fn test_complex_not_equal() {
    let t1 = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]),
        &Default::default(),
    );
    let t2 = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );

    t1.not_equal(t2)
        .into_data()
        .assert_eq(&TensorData::from([false, true]), false);
}

#[test]
fn test_complex_any() {
    TestTensor::<1>::zeros([2], &Default::default())
        .any()
        .into_data()
        .assert_eq(&TensorData::from([false]), false);

    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );
    tensor
        .any()
        .into_data()
        .assert_eq(&TensorData::from([true]), false);
}

#[test]
fn test_complex_any_dim() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                Complex::<f32> {
                    real: 0.0,
                    imag: 0.0,
                },
                Complex::<f32> {
                    real: 1.0,
                    imag: 0.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 0.0,
                    imag: 0.0,
                },
                Complex::<f32> {
                    real: 0.0,
                    imag: 0.0,
                },
            ],
        ]),
        &Default::default(),
    );

    // col 0: both zero -> false; col 1: has non-zero -> true
    tensor
        .any_dim(0)
        .into_data()
        .assert_eq(&TensorData::from([[false, true]]), false);
}

#[test]
fn test_complex_all() {
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 0.0,
                imag: 2.0,
            },
        ]),
        &Default::default(),
    );
    tensor
        .all()
        .into_data()
        .assert_eq(&TensorData::from([true]), false);

    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );
    tensor
        .all()
        .into_data()
        .assert_eq(&TensorData::from([false]), false);
}

#[test]
fn test_complex_all_dim() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                Complex::<f32> {
                    real: 1.0,
                    imag: 0.0,
                },
                Complex::<f32> {
                    real: 1.0,
                    imag: 0.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 0.0,
                    imag: 0.0,
                },
                Complex::<f32> {
                    real: 1.0,
                    imag: 0.0,
                },
            ],
        ]),
        &Default::default(),
    );

    // col 0: has zero -> false; col 1: all non-zero -> true
    tensor
        .all_dim(0)
        .into_data()
        .assert_eq(&TensorData::from([[false, true]]), false);
}

#[test]
fn test_complex_permute() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                Complex::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                Complex::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                Complex::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
        ]),
        &Default::default(),
    );

    // permute([1, 0]) on [2, 2] is equivalent to transpose
    let data = tensor.permute([1, 0]).into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 5.0,
                imag: 6.0,
            },
        ],
        [
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
            Complex::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_expand() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    // shape [1, 2] -> expand to [3, 2]
    let data = tensor.expand([3, 2]).into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
        [
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
        [
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_flip() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                Complex::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                Complex::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                Complex::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let data = tensor.flip([0]).into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 5.0,
                imag: 6.0,
            },
            Complex::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ],
        [
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_unfold() {
    // shape [4] -> unfold(0, size=2, step=1) -> shape [3, 2]
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 2.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 4.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );

    let result: TestTensor<2> = tensor.unfold(0, 2, 1);
    let data = result.into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 2.0,
                imag: 0.0,
            },
        ],
        [
            Complex::<f32> {
                real: 2.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 0.0,
            },
        ],
        [
            Complex::<f32> {
                real: 3.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 4.0,
                imag: 0.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_slice_assign() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                Complex::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                Complex::<f32> {
                    real: 2.0,
                    imag: 2.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 3.0,
                    imag: 3.0,
                },
                Complex::<f32> {
                    real: 4.0,
                    imag: 4.0,
                },
            ],
        ]),
        &Default::default(),
    );
    let values = TestTensor::<2>::from_data(
        TensorData::from([[
            Complex::<f32> {
                real: 9.0,
                imag: 9.0,
            },
            Complex::<f32> {
                real: 8.0,
                imag: 8.0,
            },
        ]]),
        &Default::default(),
    );

    let data = tensor.slice_assign([0..1, 0..2], values).into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 9.0,
                imag: 9.0,
            },
            Complex::<f32> {
                real: 8.0,
                imag: 8.0,
            },
        ],
        [
            Complex::<f32> {
                real: 3.0,
                imag: 3.0,
            },
            Complex::<f32> {
                real: 4.0,
                imag: 4.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_select() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                Complex::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                Complex::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                Complex::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 9.0,
                    imag: 10.0,
                },
                Complex::<f32> {
                    real: 11.0,
                    imag: 12.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let indices = Tensor::<TestBackend, 1, Int>::from_ints([2, 0], &Default::default());
    let data = tensor.select(0, indices).into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 9.0,
                imag: 10.0,
            },
            Complex::<f32> {
                real: 11.0,
                imag: 12.0,
            },
        ],
        [
            Complex::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}
