import argparse
import string
import textwrap
from Reaxys_API import Reaxys_API

API_URL = 'https://www.reaxys.com/reaxys/api'

cas_ids = ["64-18-6", "64-19-7", "64-67-5"]
patent_numbers = ["WO2010/59773", "WO2011/159553"]

structure_v2000 = """structure('

HDR
 14 14  0  0  1  0  0  0  0  0999 V2000
    1.6722    0.9384    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7056    1.1977    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1713    1.8045    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3300    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.4904    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.4480    2.1643    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.1715    1.8045    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.6724    2.6705    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.6724    0.9384    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.6708    2.6705    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.6708    0.9384    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.1717    1.8045    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.1736    1.8045    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.5788    0.5151    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1 14  1  1  0  0  0
  2  5  1  0  0  0  0
  2  6  2  0  0  0  0
  3  7  1  0  0  0  0
  7  8  2  0  0  0  0
  7  9  1  0  0  0  0
  8 10  1  0  0  0  0
  9 11  2  0  0  0  0
 10 12  2  0  0  0  0
 11 12  1  0  0  0  0
 12 13  1  0  0  0  0
M  REG 392441
M  END
','exact,tautomers,no_extra_rings')"""



structure_v3000 = """structure('
  Marvin  11031004132D
HDR
  0  0  0     0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 8 8 0 0 0
M  V30 BEGIN ATOM
M  V30 1 C 5.8641 -1.502 0 0
M  V30 2 C 5.8641 -2.498 0 0
M  V30 3 C 5.0058 -3.001 0 0
M  V30 4 C 4.1359 -2.498 0 0
M  V30 5 C 4.1359 -1.502 0 0
M  V30 6 C 4.9941 -0.999 0 0
M  V30 7 O 6.7333 -0.999 0 0
M  V30 8 O 3.2667 -3.001 0 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 2 1 2
M  V30 2 1 1 6
M  V30 3 1 1 7
M  V30 4 1 2 3
M  V30 5 2 3 4
M  V30 6 1 4 5
M  V30 7 1 4 8
M  V30 8 2 5 6
M  V30 END BOND
M  V30 END CTAB
M  END
','exact,isotopes,stereo_absolute,salts,mixtures,charges,radicals')"""


def example_1(ra):
    for cas_id in cas_ids:
        ra.select("RX", "S", "IDE.RN = '" + cas_id + "'", "",
                  "WORKER,NO_CORESULT")
        if ra.resultsize != 0:
            response = ra.retrieve(ra.resultname, ["MP"], str(1),
                                   str(ra.resultsize), "", "", "", "")
            results = ra.get_field_content(response, "MP.MP")
            for r in results:
                datapoint = cas_id + "\t" + r + "\n"
                print(datapoint)


def example_2(ra):
    ra.select("RX", "R", structure_v2000, "", "WORKER,NO_CORESULT")
    response = ra.retrieve(
        ra.resultname,
        ["RY"], "1", "1", "", "", "", "ISSUE_RXN=true,ISSUE_RCT=false")
    results = ra.get_field_content(response, "RY.STR")
    for i in results:
        print(i)



if __name__ == "__main__":
    examples = '''
    Examples
    1. Takes a list of CAS numbers and retrieves melting points for the
    corresponding compounds.

    2. Takes a structure and retrieves an RD file that contains a reaction,
    in which the corresponding compound appears as a starting material or a
    product.

    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(examples))
    parser.add_argument('-e', '--example', type=int, required=True)
    parser.add_argument('-c', '--caller', required=True)
    parser.add_argument('-u', '--user', default="")
    parser.add_argument('-p', '--password', default="")
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    api = Reaxys_API()

    if args.debug:
        api.debug = True

    api.connect(API_URL, '', args.user, args.password, args.caller)

    if args.example == 1:
        example_1(api)
    elif args.example == 2:
        example_2(api)

    api.disconnect()
