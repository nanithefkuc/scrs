#![cfg(feature = "internals")]
#![allow(missing_docs)]

use scrs::afft::TransformPlan;
use scrs::decoder::RecipeCache;
use scrs::internals::afft::shared_transform_plan;
use scrs::internals::decoder::RecipeCacheExt;
use scrs::{Engine, gf8, gf16};

#[test]
fn internals_feature_exposes_implementation_facades() {
    let profile = scrs::internals::codec::profile_from_parts(Engine::Afft, 4, 2, 8);
    assert_eq!(profile.engine(), Engine::Afft);

    let internal_profile = scrs::internals::afft::Profile::new(4, 2, 8).unwrap();
    assert_eq!(internal_profile.transform_size, 8);

    let plan = shared_transform_plan(4).unwrap();
    let mut rows = vec![0u8; plan.size() * 2];
    plan.forward_bytes(&mut rows, 2).unwrap();
    assert!(rows.iter().all(|&byte| byte == 0));

    let mut cache = RecipeCache::new(3);
    assert_eq!(RecipeCacheExt::capacity(&cache), 3);
    RecipeCacheExt::set_capacity(&mut cache, 4);
    assert_eq!(RecipeCacheExt::capacity(&cache), 4);

    let mut gf8_destination = [0u8; 4];
    scrs::internals::payload::xor_scaled_bytes(
        &mut gf8_destination,
        gf8::Elem::ONE,
        &[1, 2, 3, 4],
    );
    assert_eq!(gf8_destination, [1, 2, 3, 4]);

    let mut tower_values = [gf16::Elem::ONE];
    scrs::internals::tower::batch_invert(&mut tower_values);
    assert_eq!(tower_values, [gf16::Elem::ONE]);

    let direct = TransformPlan::new(4).unwrap();
    assert_eq!(direct.size(), plan.size());
}

#[test]
fn internals_feature_exposes_both_backend_layers() {
    use scrs::internals::backend::{
        Backend, backend_for, has_vector_elementwise, payload_backend, transform_backend,
    };

    // `Backend` derives `Ord` over its declaration order, which is *preference*
    // order: a stronger backend compares LESS. So "never stronger than X" is
    // `>= X`, not `<= X`.
    let payload = payload_backend();
    let transform = transform_backend();
    assert!(Backend::ALL.contains(&payload));
    assert!(Backend::ALL.contains(&transform));

    // cafft caps its butterflies at Gfni, so the transform layer is never
    // stronger than the payload layer.
    assert!(transform >= payload);
    assert!(transform >= Backend::Gfni);

    // A field's kernels never exceed what the host resolved to.
    assert!(backend_for::<fff::Gf8>() >= payload);
    assert!(backend_for::<fff::Gf16>() >= payload);

    // Reported purely so a downstream tuner can branch on it; both fields answer.
    let _ = has_vector_elementwise::<fff::Gf8>();
    let _ = has_vector_elementwise::<fff::Gf16>();
}
