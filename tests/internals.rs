#![cfg(feature = "internals")]
#![allow(missing_docs)]

use scrs::afft::TransformPlan;
use scrs::decoder::RecipeCache;
use scrs::internals::afft::{TransformPlanExt, shared_transform_plan};
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
    plan.forward_bytes(&mut rows, 2);
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
fn internals_feature_exposes_the_active_backend() {
    use scrs::internals::backend::{Backend, backend, backend_for, has_vector_elementwise};

    // Whatever the host resolves to must be a real backend, and the per-field query
    // must never claim more than the process-wide one.
    let active = backend();
    assert!(Backend::ALL.contains(&active));
    assert!(backend_for::<fff::Gf8>() <= active);
    assert!(backend_for::<fff::Gf16>() <= active);

    // Reported purely so a downstream tuner can branch on it; both fields answer.
    let _ = has_vector_elementwise::<fff::Gf8>();
    let _ = has_vector_elementwise::<fff::Gf16>();
}
