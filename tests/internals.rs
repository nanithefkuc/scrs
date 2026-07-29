//! Parity checklist for the `internals` feature.
//!
//! Every item the feature exposes is named here. A gap in this file is a gap in
//! the feature: if a path stops resolving, the facade regressed. The assertions
//! are deliberately weak where the value is an implementation detail — the point
//! is that the path *resolves* and the signature still fits.

#![cfg(feature = "internals")]

use scrs::{Engine, gf8, gf16};

/// Reachability of the `scrs::internals::*` facade proper.
#[test]
fn facade_modules_resolve() {
    let profile = scrs::internals::codec::profile_from_parts(Engine::Gf16Afft, 4, 2, 8);
    assert_eq!(profile.engine(), Engine::Gf16Afft);

    let internal_profile = scrs::internals::afft::Profile::<fff::Gf16>::new(4, 2, 8).unwrap();
    assert_eq!(internal_profile.transform_size, 8);

    let plan = scrs::internals::afft::shared_transform_plan::<fff::Gf16>(4).unwrap();
    let mut rows = vec![0u8; plan.size() * 2];
    plan.forward_bytes(&mut rows, 2).unwrap();
    assert!(rows.iter().all(|&byte| byte == 0));

    assert!(scrs::internals::afft::zeroed_bytes(4).unwrap().iter().all(|&b| b == 0));

    let mut destination = [0u8; 4];
    scrs::internals::payload::xor_scaled_bytes(&mut destination, gf8::Elem::ONE, &[1, 2, 3, 4]);
    assert_eq!(destination, [1, 2, 3, 4]);

    let mut values = [gf16::Elem::ONE];
    scrs::internals::tower::batch_invert(&mut values);
    assert_eq!(values, [gf16::Elem::ONE]);
}

/// `RecipeCache`'s fields are `pub(crate)`, so the extension trait is the only
/// way in. It is the one part of the facade a module gate cannot replace.
#[test]
fn recipe_cache_ext_reaches_private_fields() {
    use scrs::decoder::RecipeCache;
    use scrs::internals::decoder::RecipeCacheExt;

    let mut cache = RecipeCache::new(3);
    assert_eq!(RecipeCacheExt::capacity(&cache), 3);
    RecipeCacheExt::set_capacity(&mut cache, 4);
    assert_eq!(RecipeCacheExt::capacity(&cache), 4);

    assert!(RecipeCacheExt::entries(&cache).is_empty());
    assert!(RecipeCacheExt::entries_mut(&mut cache).is_empty());
    assert!(RecipeCacheExt::last(&cache).is_none());
    assert!(RecipeCacheExt::last_mut(&mut cache).is_none());

    RecipeCacheExt::reset_statistics(&mut cache);
    assert_eq!(cache.hits(), 0);
    assert_eq!(cache.misses(), 0);
}

/// Both kernel layers report a real backend, and neither may claim more than the
/// host resolved to.
#[test]
fn backend_layers_agree() {
    use scrs::internals::backend::{
        Backend, backend_for, has_vector_elementwise, payload_backend, transform_backend,
    };

    let payload = payload_backend();
    let transform = transform_backend();
    assert!(Backend::ALL.contains(&payload));
    assert!(Backend::ALL.contains(&transform));

    // `Backend` derives `Ord` over its declaration order, which is *preference*
    // order: a stronger backend compares LESS. So "never stronger than X" is
    // `>= X`. cafft caps its butterflies at Gfni.
    assert!(transform >= payload);
    assert!(transform >= Backend::Gfni);
    assert!(backend_for::<fff::Gf8>() >= payload);
    assert!(backend_for::<fff::Gf16>() >= payload);

    let _ = has_vector_elementwise::<fff::Gf8>();
    let _ = has_vector_elementwise::<fff::Gf16>();
}

/// Gated modules under `afft`, plus the locator caches and the crossover
/// threshold that decides which finalize path a decode takes.
#[test]
fn afft_module_is_reachable() {
    use scrs::afft::decoder::{
        GF8_LOCATORS, GF16_LOCATORS, TARGETED_MAX_MISSING, fit, systematic_locators,
    };

    assert!(TARGETED_MAX_MISSING > 0);
    let _ = &*GF8_LOCATORS;
    let _ = &*GF16_LOCATORS;
    let _ = systematic_locators::<fff::Gf8>();
    let _ = systematic_locators::<fff::Gf16>();

    let mut buffer = Vec::new();
    assert_eq!(fit(&mut buffer, 8).len(), 8);

    let profile = scrs::afft::profile::Profile::<fff::Gf8>::new(4, 2, 8).unwrap();
    assert_eq!(profile.evaluation_index(0), 0);
    assert!(scrs::afft::profile::zeroed_bytes(4).is_some());
}

/// The AFFT encoder's strip encoder and scratch, including cafft's
/// `encode_with_width` strip-width knob reached through `inner()`.
#[test]
fn afft_encoder_internals_expose_the_strip_width_knob() {
    use scrs::BatchEncoder;

    let (k, m, symbol_len) = (8usize, 4usize, 64usize);
    let data: Vec<u8> = (0..k * symbol_len).map(|i| (i % 251) as u8).collect();

    let encoder = scrs::afft::Gf8Encoder::new(k, m, symbol_len).unwrap();
    assert_eq!(encoder.profile().k, k);

    let mut scratch = encoder.encode_scratch();
    let mut reference = vec![0u8; m * symbol_len];
    encoder
        .encode_into_with(&data, &mut reference, &mut scratch)
        .unwrap();

    // Same repairs via cafft's width-parameterised entry point. The width is a
    // column-strip byte count and must not exceed one row.
    let mut widened = vec![0u8; m * symbol_len];
    encoder
        .inner()
        .encode_with_width(&data, &mut widened, scratch.inner_mut(), symbol_len / 2)
        .unwrap();
    assert_eq!(widened, reference, "strip width must not change the codeword");
}

/// `DecodeScratch::missing_data_mut` is the crossover-forcing knob: both
/// finalize paths read that list rather than deriving it, so a benchmark can
/// drive the targeted path and the locator path over one erasure pattern.
#[test]
fn afft_decoder_internals_force_both_finalize_paths() {
    let (k, m, symbol_len) = (8usize, 4usize, 64usize);
    let data: Vec<u8> = (0..k * symbol_len).map(|i| (i % 251) as u8).collect();

    let encoder = scrs::afft::Gf8Encoder::new(k, m, symbol_len).unwrap();
    let repairs = encoder.encode(&data).unwrap();

    let mut decoder = scrs::afft::Gf8Decoder::new(k, m, symbol_len).unwrap();
    // Drop data symbol 0; feed the rest plus one repair.
    for index in 1..k {
        decoder
            .push_symbol(index, &data[index * symbol_len..(index + 1) * symbol_len])
            .unwrap();
    }
    decoder.push_symbol(k, &repairs[0]).unwrap();

    assert!(decoder.bit(1));
    assert!(!decoder.bit(0));
    // The AFFT decoder holds payloads in the *transform* domain, so the buffer is
    // `transform_size` rows wide, not `k + m`.
    assert_eq!(
        decoder.payloads().len(),
        decoder.profile().transform_size * symbol_len
    );
    assert!(!decoder.received_bits().is_empty());
    assert_eq!(decoder.profile().k, k);
    assert_eq!(decoder.plan().size(), decoder.profile().transform_size);
    assert!(decoder.ensure_complete().is_ok());

    let mut scratch = decoder.decode_scratch();
    decoder.ensure_decode_scratch(&mut scratch).unwrap();
    assert_eq!(scratch.k(), k);
    assert_eq!(scratch.m(), m);
    assert_eq!(scratch.symbol_len(), symbol_len);
    assert_eq!(scratch.transform_size(), decoder.profile().transform_size);

    // `finalize_complete_into` is the entry point: it derives `missing_data`,
    // copies the *received* data rows into `output`, and only then dispatches on
    // `TARGETED_MAX_MISSING`. The two lower paths fill the missing rows ONLY, so
    // driving one directly requires a scratch whose `missing_data` is already
    // populated and an `output` already holding the present rows. That is the
    // contract a benchmark must honour to time the paths against each other.
    let mut complete = vec![0u8; k * symbol_len];
    decoder
        .finalize_complete_into(&mut complete, &mut scratch)
        .unwrap();
    assert_eq!(complete, data, "dispatching path must reconstruct");
    assert_eq!(
        scratch.missing_data(),
        &[0],
        "one erasure, so `complete` dispatched to the targeted path"
    );
    assert!(scratch.missing_data().len() <= scrs::afft::decoder::TARGETED_MAX_MISSING);

    let mut targeted = complete.clone();
    decoder
        .finalize_targeted_into(&mut targeted, &mut scratch)
        .unwrap();
    assert_eq!(targeted, data, "targeted path must reconstruct");

    // Same erasure pattern through the domain-sized path the dispatcher did not
    // choose: this is the crossover measurement the feature exists to enable.
    let mut locator = complete.clone();
    decoder
        .finalize_locator_into(&mut locator, &mut scratch)
        .unwrap();
    assert_eq!(locator, data, "locator path must agree with the targeted one");

    // Scratch inspection: every buffer the paths share.
    assert!(!scratch.known().is_empty());
    assert!(!scratch.recovered().is_empty());
    let _ = scratch.locator();
    let _ = scratch.locator_scratch();
    let _ = scratch.recovery();
    let _ = scratch.repair_indices();
    let _ = scratch.generator();
    let _ = scratch.system();
    let _ = scratch.inverse();
    let _ = scratch.augmented();
    let _ = scratch.coefficients();
    let _ = scratch.residuals();
    let _ = decoder.systematic_locator();
    let _ = decoder.systematic_locator_cache();
    assert_eq!(scratch.missing_data_mut().as_slice(), &[0]);
}

/// The Cauchy streaming decoder's recipe machinery and its source bound.
#[test]
fn cauchy_decoder_internals_are_reachable() {
    use scrs::decoder::streaming::MAX_SOURCES;

    assert!(MAX_SOURCES >= 256, "GF(256) codewords reach n = k + m = 256");

    let (k, m, symbol_len) = (8usize, 4usize, 64usize);
    let data: Vec<u8> = (0..k * symbol_len).map(|i| (i % 251) as u8).collect();

    let codec = scrs::batch::GoodCauchyBatchCodec::new(k, m, symbol_len).unwrap();
    let mut repairs = vec![0u8; m * symbol_len];
    codec.encode_into(&data, &mut repairs).unwrap();

    let mut decoder =
        scrs::decoder::LazyDecoderState::<scrs::good_cauchy::GoodCauchyView>::new(k, m, symbol_len)
            .unwrap();
    for index in 1..k {
        scrs::stream::SymbolSink::push(
            &mut decoder,
            index,
            &data[index * symbol_len..(index + 1) * symbol_len],
        )
        .unwrap();
    }
    scrs::stream::SymbolSink::push(&mut decoder, k, &repairs[..symbol_len]).unwrap();

    assert!(decoder.ensure_complete().is_ok());
    let recipe = decoder.build_recipe().unwrap();
    assert!(recipe.allocated_bytes() > 0);

    let mut out = vec![0u8; k * symbol_len];
    decoder.apply_recipe_into(&recipe, &mut out);
    assert_eq!(out, data, "recipe application must reconstruct");

    let mut cache = scrs::decoder::RecipeCache::new(2);
    let cached = decoder.recipe_from_cache(&mut cache).unwrap();
    assert_eq!(cached.allocated_bytes(), recipe.allocated_bytes());

    let _ = decoder.cauchy();
    assert_eq!(decoder.payloads().len(), (k + m) * symbol_len);
    let _ = decoder.staging();
}

/// Cauchy inverse helpers and the batch codec's elimination kernel.
#[test]
fn cauchy_inverse_and_batch_internals_are_reachable() {
    use scrs::decoder::cauchy_inverse::{batch_invert, cauchy_inverse_closed_form};

    let mut values = [gf8::Elem::ONE, gf8::Elem::from_raw(2)];
    batch_invert(&mut values);
    assert_eq!(values[0], gf8::Elem::ONE);

    let rows = [gf8::Elem::from_raw(1), gf8::Elem::from_raw(2)];
    let cols = [gf8::Elem::from_raw(3), gf8::Elem::from_raw(4)];
    assert_eq!(cauchy_inverse_closed_form(&rows, &cols).len(), 4);

    // A 2x2 identity inverts to itself.
    let mut matrix = [
        gf8::Elem::ONE,
        gf8::Elem::ZERO,
        gf8::Elem::ZERO,
        gf8::Elem::ONE,
    ];
    let mut inverse = vec![gf8::Elem::ZERO; 4];
    assert!(scrs::batch::codec::invert_square_into(
        &mut matrix,
        2,
        &mut inverse
    ));
    assert_eq!(inverse[0], gf8::Elem::ONE);
    assert_eq!(inverse[1], gf8::Elem::ZERO);
}

/// Tower Cauchy internals: the generator power ladder, encoder state, and the
/// decode scratch's coefficient buffers.
#[test]
fn tower_internals_are_reachable() {
    use scrs::tower::cauchy::power;

    assert_eq!(power(0), gf16::Elem::ONE);

    let (k, m, symbol_len) = (4usize, 2usize, 8usize);
    let data: Vec<Vec<u8>> = (0..k)
        .map(|i| (0..symbol_len).map(|j| (i * symbol_len + j) as u8).collect())
        .collect();

    let mut encoder = scrs::tower::StreamingEncoder::new(k, m, symbol_len).unwrap();
    for (index, symbol) in data.iter().enumerate() {
        encoder.feed_data_symbol(index, symbol).unwrap();
    }
    assert_eq!(encoder.coefficients().len(), k * m);
    assert_eq!(encoder.repairs().len(), m * symbol_len);
    assert_eq!(encoder.fed().len(), k);
    assert!(encoder.fed().iter().all(|&f| f));

    let repair = encoder.repair_symbol(0).unwrap().to_vec();
    assert!(scrs::tower::encoder::zeroed_bytes(4).is_some());
    assert!(scrs::tower::decoder::zeroed_bytes(4).is_some());

    let mut decoder = scrs::tower::LazyDecoderState::new(k, m, symbol_len).unwrap();
    for index in 1..k {
        decoder.push_symbol(index, &data[index]).unwrap();
    }
    decoder.push_symbol(k, &repair).unwrap();

    assert!(decoder.bit(1));
    assert!(!decoder.bit(0));
    assert!(decoder.ensure_complete().is_ok());
    let _ = decoder.cauchy();
    assert_eq!(decoder.payloads().len(), (k + m) * symbol_len);
    assert!(!decoder.received_bits().is_empty());

    let mut scratch = scrs::tower::decoder::DecodeScratch::new(k, m, symbol_len);
    assert_eq!(scratch.k(), k);
    assert_eq!(scratch.m(), m);
    assert_eq!(scratch.symbol_len(), symbol_len);

    decoder.build_recipe_into(&mut scratch).unwrap();
    assert_eq!(scratch.missing_data(), &[0]);
    assert_eq!(scratch.present_data().len(), k - 1);
    assert_eq!(scratch.repair_columns().len(), 1);

    let mut out = vec![0u8; k * symbol_len];
    decoder.apply_recipe_into(&scratch, &mut out);
    assert_eq!(out, data.concat(), "tower recipe must reconstruct");

    let mut direct = vec![0u8; k * symbol_len];
    decoder
        .finalize_into_with_scratch(&mut direct, &mut scratch)
        .unwrap();
    assert_eq!(direct, data.concat());

    // Every coefficient buffer the reduced solve fills.
    let _ = scratch.row_variables();
    let _ = scratch.column_variables();
    let _ = scratch.present_variables();
    let _ = scratch.row_cross();
    let _ = scratch.column_cross();
    let _ = scratch.reciprocals();
    let _ = scratch.inversion_prefixes();
    let _ = scratch.row_factors();
    let _ = scratch.column_factors();
    let _ = scratch.inverse();
    let _ = scratch.present_coefficients();
    let _ = scratch.prefix();
    let _ = scratch.suffix();
    let _ = scratch.source_indices();
    let _ = scratch.source_coefficients();
}

/// The GF(256) incremental encoder's coefficient matrix and feed state.
#[test]
fn incremental_encoder_internals_are_reachable() {
    let (k, m, symbol_len) = (4usize, 2usize, 8usize);
    let mut encoder = scrs::encoder::StreamingEncoder::new(k, m, symbol_len).unwrap();
    assert_eq!(encoder.coeffs().len(), k * m);
    assert_eq!(encoder.repairs().len(), m * symbol_len);
    assert_eq!(encoder.fed().len(), k);
    assert!(encoder.fed().iter().all(|&f| !f));

    encoder.feed_data_symbol(0, &vec![1u8; symbol_len]).unwrap();
    assert!(encoder.fed()[0]);
}

/// Matrix and selector internals, reached as methods and free items on
/// already-public types rather than through the facade.
#[test]
fn matrix_and_selector_internals_are_reachable() {
    use scrs::cauchy::{CauchyView, combinations};
    use scrs::good_cauchy::{EXP, GoodCauchyView, exp};

    assert_eq!(scrs::selector::engine_capacity(Engine::GoodCauchy), 255);
    assert_eq!(scrs::selector::engine_capacity(Engine::StandardCauchy), 256);

    let cauchy = CauchyView::new(4, 2).unwrap();
    // Standard Cauchy splits the field: x from the low half, y from the high.
    assert_ne!(cauchy.x_at(0), cauchy.y_at(0));

    let good = GoodCauchyView::new(4, 2).unwrap();
    assert_ne!(good.x_at(0), good.y_at(0));

    assert_eq!(EXP.len(), 255);
    assert_eq!(exp(0), 1, "g^0 == 1");

    assert_eq!(combinations(3, 2).count(), 3);

    let buffer = vec![gf8::Elem::ZERO; 4];
    let view = scrs::matrices::MatrixView::new(&buffer, 2, 2).unwrap();
    assert_eq!(view.buf().len(), 4);
    assert_eq!(view.rows(), 2);
    assert_eq!(view.cols(), 2);

    let mut buffer_mut = vec![gf8::Elem::ZERO; 4];
    let view_mut = scrs::matrices::MatrixViewMut::new(&mut buffer_mut, 2, 2).unwrap();
    assert_eq!(view_mut.buf().len(), 4);
}
