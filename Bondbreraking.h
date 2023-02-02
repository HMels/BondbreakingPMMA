#ifndef __BONDBREAKING_H_
#define __BONDBREAKING_H_
#include <functional>
#include "../core/particle.h"
#include "../common/util/table_1D.h"
#include "../common/util/table_2D.h"
#include "../common/util/table_3D.h"
#include "units.h"

namespace nbl { 
    namespace scatter {
        /**
        * \brief Inelastic scattering, according to the full Penn Model.
        *
        * We distinguish between inner-shell and Fermi-sea excitations. Inner-shells
        * are those with binding > 50 eV
        * − Fermi sea : same as Mao et al , doi :10.1063/1.3033564
        * − Inner shells: same as Kieft et al , doi :10.1088/0022−3727/41/21/215310
        *                 ( see also ::nbl::scatter::inelastic_thomas)
        *
        * \tparam gpu_flag Is the code to be run on a GPU?
        * \tparam opt_optical_phonon_loss           Assume optical phonon loss for energy loss less than band gap
        * \tparam opt_generate_secondary            Generate secondary electrons
        * \tparam opt_instantaneous_momentum        Large losses: consider instantaneous momentum for SE
        * \tparam opt_momentum_conservation         Large losses: obey conservation of momentum
        *
        */

        template<bool gpu_flag ,
                    bool opt_optical_phonon_loss = true ,
                    bool opt_generate_secondary = true ,
                    bool opt_instantaneous_momentum = true ,
                    bool opt_momentum_conservation = true
                    >
        class bondbreaking {
        public:
            /* *
            * \brief Indicate when this class generates secondary electrons
            */
            constexpr static bool may_create_se = opt_generate_secondary;

            /* *
            * \brief Sample a random free path length
            */
            inline PHYSICS real sample_path(particle const &this_particle,
                                            util::random_generator<gpu_flag> &rng) const {
                if (this_particle.kin_energy > Bind) // particle must be able to break the bond
                {
                    // Get inverse mean free path for this kinetic energy
                    real imfp; // total inverse mean free path as calculated in my paper
                    {
                        const real number_density = 28.2;
                        const real prefactor1 = 1 / (eps0_4_pi * eps0_4_pi);

                        const real prefactor2 = pi * pow(e, 4) / (this_particle.kin_energy + 2 * Bind);
                        const real Phi = cosr(sqrtr(Ryd / (this_particle.kin_energy + Bind)) * logr(this_particle.kin_energy / Bind));

                        const real vriens1 = (5 / (3 * Bind)) - 1 / this_particle.kin_energy - (2 * Bind / (3 * this_particle.kin_energy * this_particle.kin_energy)) - (Phi / (this_particle.kin_energy + Bind)) * logr((this_particle.kin_energy / Bind));

                        const real prefactor12 = prefactor1 * prefactor2;

                        imfp = number_density * prefactor12 * vriens1;
                    }
                    // Draw a distance
                    return rng.exponential(1 / imfp);
                } 
                else {
                    return std::numeric_limits<real>::infinity();
                }
            }

            /* *
            * \brief Perform a scattering event
            */
            template<typename particle_manager>
            inline PHYSICS void execute(
                particle_manager &particle_mgr,
                typename particle_manager::particle_index_tp particle_idx,
                util::random_generator<gpu_flag> &rng) const {
                // Retrieve current particle from global memory
                auto this_particle = particle_mgr[particle_idx];
                real omega; // the energy transfer is chosen from the distribution calculated in my paper
                {
                    const real x = logr(this_particle.kin_energy);
                    const real y = rng.unit();
                    omega = expr(_log_omega_icdf_table.get(x, y));
                }
                // Normalised direction
                this_particle.dir = normalised(this_particle.dir);
                // Deflect primary
                const vec3 prim_normal_dir = normalised(make_normal_vec(this_particle.dir, rng.phi()));
                // Random deflection (Page 80 T. Verduin)
                const real costheta_pi_pf = sqrtr(1 - ((omega + Bind) / (this_particle.kin_energy + 2 * Bind)));
                const real sintheta_pi_pf = sqrtr(1 - costheta_pi_pf * costheta_pi_pf);

                this_particle.kin_energy -= (omega);
                this_particle.dir = this_particle.dir * costheta_pi_pf + prim_normal_dir * sintheta_pi_pf;

                particle_mgr[particle_idx] = this_particle;
                if (opt_generate_secondary) {
                    particle secondary_particle;
                    secondary_particle.pos = this_particle.pos;
                    const vec3 prim_normal_dir = normalised(make_normal_vec(this_particle.dir, rng.phi()));
                    // random deflection
                    // page 80 T. Verduin
                    const real costheta_sec_pf = sqrtr((omega+Bind) / (this_particle.kin_energy + 2*Bind));
                    const real sintheta_sec_pf = sqrtr(1 - costheta_sec_pf * costheta_sec_pf);

                    this_particle.kin_energy += (omega-Bind);
                    this_particle.dir = this_particle.dir * costheta_sec_pf + prim_normal_dir * sintheta_sec_pf;

    particle_mgr.create_secondary(particle_idx, secondary_particle);
  }
  return;
}
                      

/* *
* \brief Create, given a legacy material file: this function immediately
* throws an exception.
*
* \deprecated Old file format is deprecated and not supported by all
* scattering mechanisms. This function will be removed soon.
*/
static CPU bondbreaking create(material_legacy_thomas const& mat) {
  throw std::runtime_error("Cannot get Penelope inelastic model from old-style .mat files");
}

/* *
* \brief Create, given a material file.
*/
static CPU bondbreaking create(hdf5_file const& mat) {
  auto __logspace_K_at = [&](int x, int cnt) {
    return K_min * std::exp(1.0 * x / (cnt - 1)
      * std::log(K_max / K_min));
  };
  auto __linspace_P_at = [&](int y, int cnt) {
    return 1.0 * y / (cnt - 1);
  };
  bondbreaking inel;

  inel._log_omega_icdf_table = util::table_2D<real, gpu_flag>::create(logr(K_min), logr(K_max), K_cnt, 0, 1, P_cnt);
  inel._log_omega_icdf_table.mem_scope([&](real** icdf_vector) {
    auto inelastic_icdf = mat.get_table_axes<2>("bondbreaking/omega_icdf");
    for (int y = 0; y < P_cnt; ++y) {
      const units::quantity<double> P = __linspace_P_at(y, P_cnt) * units::dimensionless;

      for (int x = 0; x < K_cnt; ++x) {
        const units::quantity<double> K = __logspace_K_at(x, K_cnt) * units::eV;

        icdf_vector[y][x] = (real)std::log(inelastic_icdf.get_linear(K, P) / units::eV);
      }
    }
  });
  return inel;
}

/* *
* \brief Deallocate data held by an instance of this class.
*/
static CPU void destroy(bondbreaking& inel) {
  util::table_2D<real, gpu_flag>::destroy(inel._log_omega_icdf_table);
}

private:
      /* *
      * Table storing the energy loss values (omega) based on Vriens cross section for bondbreaking
      * created using the cstool
      */
    util::table_2D<real, gpu_flag> _log_omega_icdf_table;
};
}}  // namespace nbl::scatter
#endif
