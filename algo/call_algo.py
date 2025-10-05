# import all algorithms this benchmark implement

def call_algo(algo_name, config, mode, device):
    if mode == -1:
        algo_name = algo_name.lower()
        assert algo_name == 'sac_target_only'
        # online learning on target env for baseline
        from online_online.sac_target_only import SAC_Target_Only
        policy = SAC_Target_Only(config, device)
    elif mode == 0:
        algo_name = algo_name.lower()
        assert algo_name in ['sac', 'darc', 'vgdf', 'sac_iw', 'par', 'sac_liberty', 'darc_liberty', 'larc', 'epic', 'nomad', 'nomad_v2', 'nomad_v3', 'nomad_v4','nomad_weight']
        # online online setting
        from online_online.darc import DARC
        from online_online.sac import SAC
        from online_online.vgdf import VGDF
        from online_online.sac_iw import SAC_IW
        from online_online.par import PAR
        from online_online.sac_liberty import SAC_LIBERTY
        from online_online.darc_liberty import DARC_LIBERTY
        from online_online.larc import LARC
        from online_online.epic import EPIC
        from online_online.nomad import NOMAD
        from online_online.nomad_v2 import NOMAD_V2
        from online_online.nomad_v3 import NOMAD_V3
        from online_online.nomad_v4 import NOMAD_V4
        from online_online.nomad_weight import NOMAD_V3 as NOMAD_WEIGHT

        algo_to_call = {
            'sac': SAC,
            'darc': DARC,
            'vgdf': VGDF,
            'sac_iw': SAC_IW,
            'par': PAR,
            'sac_liberty': SAC_LIBERTY,
            'darc_liberty': DARC_LIBERTY,
            'larc': LARC,
            'epic': EPIC,
            'nomad': NOMAD,
            'nomad_v2': NOMAD_V2,
            'nomad_v3': NOMAD_V3,
            'nomad_v4': NOMAD_V4,
            'nomad_weight': NOMAD_WEIGHT,
        }

        algo = algo_to_call[algo_name]
        policy = algo(config, device)

    elif mode == 1:
        algo_name = algo_name.lower()
        assert algo_name in ['cql_sac', 'bc_vgdf', 'bc_sac', 'h2o', 'mcq_sac', 'rlpd', 'bc_par']
        # offline online setting
        from offline_online.cql_sac import CQLSAC
        from offline_online.bc_vgdf import BCVGDF
        from offline_online.bc_sac import BCSAC
        from offline_online.mcq_sac import MCQSAC
        from offline_online.h2o import H2O
        from offline_online.rlpd import RLPD
        from offline_online.bc_par import BCPAR

        algo_to_call = {
            'cql_sac': CQLSAC,
            'bc_vgdf': BCVGDF,
            'bc_sac': BCSAC,
            'mcq_sac': MCQSAC,
            'h2o': H2O,
            'rlpd': RLPD,
            'bc_par': BCPAR,
        }

        algo = algo_to_call[algo_name]
        policy = algo(config, device)

    elif mode == 2:
        algo_name = algo_name.lower()
        assert algo_name in ['sac_bc', 'h2o', 'sac_cql', 'sac_mcq', 'par_bc']
        # online offline setting
        from online_offline.sac_bc import SACBC
        from online_offline.h2o import H2O
        from online_offline.sac_cql import SACCQL
        from online_offline.sac_mcq import SACMCQ
        from online_offline.par_bc import PARBC

        algo_to_call = {
            'sac_bc': SACBC,
            'h2o': H2O,
            'sac_cql': SACCQL,
            'sac_mcq': SACMCQ,
            'par_bc': PARBC,
        }
        algo = algo_to_call[algo_name]
        policy = algo(config, device)

    elif mode == 3:
        algo_name = algo_name.lower()
        assert algo_name in ['dara', 'bosa', 'iql', 'td3_bc', 'igdf']
        # offline offline setting
        from offline_offline.dara import DARA
        from offline_offline.bosa import BOSA
        from offline_offline.iql import IQL
        from offline_offline.td3_bc import TD3BC
        from offline_offline.igdf import IGDF

        algo_to_call = {
            'dara': DARA,
            'bosa': BOSA,
            'iql': IQL,
            'td3_bc': TD3BC,
            'igdf': IGDF,
        }

        algo = algo_to_call[algo_name]
        policy = algo(config, device)

    return policy