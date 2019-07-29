/**
 * C++ code for sorting WA entries.
 */

#include <stdint.h>

#include <unordered_map>
#include <unordered_set>
#include <type_traits>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#ifdef __GNUC__
typedef double dbl __attribute__((aligned()));
typedef dbl* __restrict dbls __attribute__((aligned()));

typedef intptr_t __restrict intp __attribute__((aligned()));
typedef intp* __restrict intps __attribute__((aligned()));

typedef uint8_t __restrict byte __attribute__((aligned()));
typedef byte* __restrict bytes __attribute__((aligned()));
#else
typedef double dbl;
typedef dbl* __restrict dbls;

typedef intptr_t __restrict intp;
typedef intp* __restrict intps;

typedef uint8_t __restrict byte;
typedef byte* __restrict bytes;
#endif

typedef std::unordered_map<intptr_t, std::unordered_set<intptr_t>> uv_map;

template <int NF=0, bool COUNT_FAILS=false, bool STABLE=false>
class CompN {
    const int nlvls, nfltrs;
    intps* coords; // nlvls x size (pre-dec)            u* = coords[j][u]
    dbls* thetas;  // nlvls x (size (post-dec))*nfltrs  |theta_uij| => thetas[j][u*,i]
    bytes* tw0s;   // nlvls x (size (pre-dec))*nfltrs   theta_uij*w_uij_u>0 => tw0s[j][u,i]

    // This map is only available if we are counting failures
    typedef typename std::conditional<COUNT_FAILS, uv_map&, unsigned char>::type uv_map_t;
    volatile uv_map_t fails;

    template<bool _CF=COUNT_FAILS>
    inline typename std::enable_if<_CF>::type insert_fail(intp u, intp v) const {
        fails[u].insert(v);
    }
    template<bool _CF=COUNT_FAILS>
    inline typename std::enable_if<!_CF>::type insert_fail(intp u, intp v) const {}

public:
    inline CompN(int nlvls, intps* coords, dbls* thetas, bytes* tw0s) :
        nlvls(nlvls), nfltrs(NF), coords(coords), thetas(thetas), tw0s(tw0s)
    { assert(NF != 0); }

    inline CompN(int nlvls, int nfltrs, intps* coords, dbls* thetas, bytes* tw0s) :
        nlvls(nlvls), nfltrs(nfltrs), coords(coords), thetas(thetas), tw0s(tw0s)
    { assert(NF == 0 || NF == nfltrs); }

    inline CompN(int nlvls, intps* coords, dbls* thetas, bytes* tw0s, uv_map_t fails) :
        nlvls(nlvls), nfltrs(NF), coords(coords), thetas(thetas), tw0s(tw0s), fails(fails)
    { assert(NF != 0); }

    inline CompN(int nlvls, int nfltrs, intps* coords, dbls* thetas, bytes* tw0s, uv_map_t fails) :
        nlvls(nlvls), nfltrs(nfltrs), coords(coords), thetas(thetas), tw0s(tw0s), fails(fails)
    { assert(NF == 0 || NF == nfltrs); }

    /*
     Called to compare pixel u and v. Returns true if u < v. Returns false if u > v or if they are
     not comparable. This really should be sorted as a poset using the algorithms in
     https://arxiv.org/abs/0707.1532 but that is way more complicated.
     */
    inline bool operator()(intp u, intp v) const {
        // Otherwise go through all sorted coefficients and calculate order
        int nfilters = NF ? NF : this->nfltrs; // this is solved at compile-time
        intp u_orig = u, v_orig = v;
        for (int j = 0; j < this->nlvls; j++) { // j is the filter level
            // Get the decimated and filter coordinates for the pixel at this level
            intp u_dec = this->coords[j][u], v_dec = this->coords[j][v];
            if (unlikely(u_dec == v_dec)) {
                // Not directly comparable, but if we assume that there is some |theta_vij| smaller
                // than both than all we have to do is find the first theta_uij*w_uij_u>0 that
                // aren't equal between u and v and use them to sort the values.
                bytes tw0_u = &tw0s[j][u*nfilters];
                bytes tw0_v = &tw0s[j][v*nfilters];
                for (int i = 0; i < nfilters; ++i) {
                    byte tw0_u_i = tw0_u[i], tw0_v_i = tw0_v[i];
                    if (tw0_u_i != tw0_v_i) { return tw0_u_i < tw0_v_i; }
                }
                // The above works as long as the w_uij_u used to distinguish the two is lined up
                // with a theta_uij that isn't 0. If it is then we can't find a solution still (this
                // also is because our assumption that there is a |theta_vij| smaller isn't correct
                // since there can't be any smaller values than 0).
                // No additional levels will help distinguish these since they will have the same u
                // and v at any further levels.
                insert_fail(u_orig, v_orig);
                return STABLE ? u_orig < v_orig : false;
            }

            // The |theta_uij| and |theta_vij| values for all i
            dbls theta_u = &thetas[j][u_dec*nfilters];
            dbls theta_v = &thetas[j][v_dec*nfilters];
            for (int i = 0; i < nfilters; ++i) {
                // Check for ordering
                dbl theta_u_i = theta_u[i], theta_v_i = theta_v[i];
                if (theta_u_i > theta_v_i) { return !tw0s[j][u*nfilters + i]; }
                if (theta_u_i < theta_v_i) { return  tw0s[j][v*nfilters + i]; }
            }

            // Update the coordinates
            u = u_dec;
            v = v_dec;
        }

        // Not enough data to make a decision (more levels could help distinguish these)
        insert_fail(u_orig, v_orig);
        return STABLE ? u_orig < v_orig : false;
    }
};

typedef CompN<0, false> Comp;
typedef CompN<4, false> Comp4;
typedef CompN<8, false> Comp8;
typedef CompN<0, true> CompFC;
typedef CompN<4, true> Comp4FC;
typedef CompN<8, true> Comp8FC;
