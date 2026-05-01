use pumpkin_data::dimension::Dimension;

use crate::ProtoChunk;
use crate::generation::generator::VanillaGenerator;
use crate::generation::proto_chunk::GenerationCache;
use crate::world::BlockRegistryExt;
use pumpkin_config::lighting::LightingEngineConfig;

use super::{Cache, Chunk, StagedChunkEnum};

fn accumulate_dependency_rings(
    stage: StagedChunkEnum,
    offset: usize,
    rings: &mut Vec<StagedChunkEnum>,
) {
    if rings.len() <= offset {
        rings.resize(offset + 1, StagedChunkEnum::None);
    }
    rings[offset] = rings[offset].max(stage);

    if stage <= StagedChunkEnum::Empty {
        return;
    }

    let dependencies = stage.get_direct_dependencies();
    for dep_radius in 0..=stage.get_direct_radius() as usize {
        accumulate_dependency_rings(dependencies[dep_radius], offset + dep_radius, rings);
    }
}

fn dependency_rings(stage: StagedChunkEnum) -> Vec<StagedChunkEnum> {
    let mut rings = Vec::new();
    accumulate_dependency_rings(stage, 0, &mut rings);
    rings
}

fn prepare_proto_chunk_to_stage(
    proto_chunk: &mut ProtoChunk,
    target_stage: StagedChunkEnum,
    generator: &VanillaGenerator,
) {
    let target_stage = target_stage.min(StagedChunkEnum::Carvers);

    if proto_chunk.stage < StagedChunkEnum::StructureStart
        && target_stage >= StagedChunkEnum::StructureStart
    {
        proto_chunk.set_structure_starts(generator);
    }

    if proto_chunk.stage < StagedChunkEnum::StructureReferences
        && target_stage >= StagedChunkEnum::StructureReferences
    {
        proto_chunk.set_structure_references(generator);
    }

    if proto_chunk.stage < StagedChunkEnum::Biomes && target_stage >= StagedChunkEnum::Biomes {
        proto_chunk.step_to_biomes(generator);
    }

    if proto_chunk.stage < StagedChunkEnum::Noise && target_stage >= StagedChunkEnum::Noise {
        proto_chunk.step_to_noise(generator);
    }

    if proto_chunk.stage < StagedChunkEnum::Surface && target_stage >= StagedChunkEnum::Surface {
        proto_chunk.step_to_surface(generator);
    }

    if proto_chunk.stage < StagedChunkEnum::Carvers && target_stage >= StagedChunkEnum::Carvers {
        proto_chunk.step_to_carvers(generator);
    }
}

pub fn generate_single_chunk(
    _dimension: &Dimension,
    _biome_mixer_seed: i64,
    generator: &VanillaGenerator,
    block_registry: &dyn BlockRegistryExt,
    chunk_x: i32,
    chunk_z: i32,
    target_stage: StagedChunkEnum,
) -> Chunk {
    let rings = dependency_rings(target_stage);
    let radius = (rings.len() as i32).saturating_sub(1);

    let mut cache = Cache::new(chunk_x - radius, chunk_z - radius, radius * 2 + 1);

    for dx in -radius..=radius {
        for dz in -radius..=radius {
            let new_x = chunk_x + dx;
            let new_z = chunk_z + dz;

            let proto_chunk = Box::new(ProtoChunk::new(new_x, new_z, generator));

            cache.chunks.push(Chunk::Proto(proto_chunk));
        }
    }

    for dx in -radius..=radius {
        for dz in -radius..=radius {
            let ring = dx.abs().max(dz.abs()) as usize;
            let required_stage = rings
                .get(ring)
                .copied()
                .unwrap_or(StagedChunkEnum::None)
                .min(StagedChunkEnum::Carvers);

            if required_stage == StagedChunkEnum::None {
                continue;
            }

            let chunk = cache
                .get_chunk_mut(chunk_x + dx, chunk_z + dz)
                .expect("generated cache must include dependency chunk");
            prepare_proto_chunk_to_stage(chunk, required_stage, generator);
        }
    }

    let stages = [
        StagedChunkEnum::Features,
        StagedChunkEnum::Lighting,
        StagedChunkEnum::Full,
    ];

    for &stage in &stages {
        if stage as u8 <= StagedChunkEnum::Surface as u8 || stage as u8 > target_stage as u8 {
            continue;
        }

        if stage as u8 > target_stage as u8 {
            break;
        }

        cache.advance(
            stage,
            generator,
            block_registry,
            &LightingEngineConfig::Default,
        );
    }

    let mid = ((cache.size * cache.size) >> 1) as usize;
    cache.chunks.swap_remove(mid)
}

#[cfg(test)]
mod tests {
    use super::{dependency_rings, prepare_proto_chunk_to_stage};
    use crate::ProtoChunk;
    use crate::biome::hash_seed;
    use crate::chunk_system::{Cache, Chunk, StagedChunkEnum, generate_single_chunk};
    use crate::generation::get_world_gen;
    use crate::generation::proto_chunk::GenerationCache;
    use crate::world::BlockRegistryExt;
    use pumpkin_data::chunk_gen_settings::GenerationSettings;
    use pumpkin_data::dimension::Dimension;
    use pumpkin_util::world_seed::Seed;
    use std::sync::Arc;

    struct BlockRegistry;
    impl BlockRegistryExt for BlockRegistry {
        fn can_place_at(
            &self,
            _block: &pumpkin_data::Block,
            _state: &pumpkin_data::BlockState,
            _block_accessor: &dyn crate::world::BlockAccessor,
            _block_pos: &pumpkin_util::math::position::BlockPos,
        ) -> bool {
            true
        }
    }

    #[test]
    fn generate_chunk_should_return() {
        let dimension = Dimension::OVERWORLD;
        let seed = Seed(42);
        let block_registry = Arc::new(BlockRegistry);
        let world_gen = get_world_gen(seed, dimension);
        let biome_mixer_seed = hash_seed(world_gen.random_config.seed);

        let _ = generate_single_chunk(
            &dimension,
            biome_mixer_seed,
            &world_gen,
            block_registry.as_ref(),
            0,
            0,
            StagedChunkEnum::Full,
        );
    }

    #[test]
    fn full_stage_dependency_rings_match_runtime_schedule() {
        assert_eq!(
            dependency_rings(StagedChunkEnum::Full),
            vec![
                StagedChunkEnum::Full,
                StagedChunkEnum::Lighting,
                StagedChunkEnum::Features,
                StagedChunkEnum::Carvers,
                StagedChunkEnum::Surface,
            ]
        );
    }

    #[test]
    fn single_chunk_dependencies_are_prepared_before_features() {
        let dimension = Dimension::OVERWORLD;
        let seed = Seed(42);
        let world_gen = get_world_gen(seed, dimension);
        let biome_mixer_seed = hash_seed(world_gen.random_config.seed);
        let rings = dependency_rings(StagedChunkEnum::Full);
        let radius = (rings.len() as i32).saturating_sub(1);

        let mut cache = Cache::new(-radius, -radius, radius * 2 + 1);
        for dx in -radius..=radius {
            for dz in -radius..=radius {
                cache
                    .chunks
                    .push(Chunk::Proto(Box::new(ProtoChunk::new(dx, dz, &world_gen))));
            }
        }

        for dx in -radius..=radius {
            for dz in -radius..=radius {
                let ring = dx.abs().max(dz.abs()) as usize;
                let target = rings[ring].min(StagedChunkEnum::Carvers);
                let chunk = cache.get_chunk_mut(dx, dz).unwrap();
                prepare_proto_chunk_to_stage(chunk, target, &world_gen);
            }
        }

        for dx in -radius..=radius {
            for dz in -radius..=radius {
                let chunk = cache.get_chunk(dx, dz).unwrap();
                let expected = if dx.abs().max(dz.abs()) == radius {
                    StagedChunkEnum::Surface
                } else {
                    StagedChunkEnum::Carvers
                };
                assert_eq!(
                    chunk.stage, expected,
                    "dependency chunk at ({dx}, {dz}) was not prepared to the expected stage"
                );
            }
        }
    }
}
