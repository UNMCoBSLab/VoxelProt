COFACTOR = ["ADP","FAD","SAM","HEM","NAD","COA","NAP"] 
"""
ADP-338
FAD-193
SAM-162
HEM-142
COA-101
NAD-92
NAP-81
"""

EXCLUDED_LIGANDS = ['CMX','TXE','01K','9LF','A2R','UJJ','3OD','F5D','5ZV','WC8','52O','KWM','KXG','OXT','5FA','NHW','WCA','ZD4','HCO',
'CA8','NDX','OXK','CCH','TUY','1OF','AQH','5TW','0T1','NDW','MC0','HFQ','HQG','BY3','BCO','MD9','NAX','BA7','HFM','UTA',
'XZQ','UFE','3VV','XRK','NDO','B7X','6YU','5J8','8TQ','YGZ','Q2C','93M','HEC','AVV','B4P','3CP','2PF','FAB','TDT','ZP4',
'HD6','SPF','HP5','ZOZ','ADQ','LNC','5I7','9NY','TKC','EE1','OH9','GEK','WC5','FAJ','Z5A','6KB','FAE','5ZY','CCQ','1O4',
'WI6','KGA','COD','ZSF','OZV','HAX','7L1','5IA','CXC','CAO','Q1X','BSJ','LHQ','IDP','8JD','AHZ','6L0','NJS','ABP','ZSI',
'NA7','ACO','AV1','6AD','HDC','CO8','COO','COF','YE2','COT','L3W','1FH','COZ','DDH','DVN','94Q','CIC','LA8','HUF',
'PAD','PAX','NBS','Q5B','NHM','DCA','YXS','MDE','128','D70','48H','FDE','GE0','OFN','9X8','AD9','NXX','9JM','UT7','DQV',
'2KQ','2MC','KGJ','BCA','6CQ','MRR','JNT','N01','5TL','ZSL','Y66','ISW','3AA','UP5','OMR','TXP','MCD','KGP','ZF9','3H9',
'AOV','ZIE','KFV','AMX','T3D','CAJ','A1R','NBD','NAQ','Q1F','PKZ','SOP','2RW','A1S','NFD','3AN','ZND','A4P','1VU','CNA',
'D73','FWC','COH','D52','XYQ','APR','80F','D3U','LYX','NA0','HEV','EAD','LCV','H6Y','522','RUR','4R2','NDC','CX0','8NA',
'FED','FNK','2NF','6V0','FAM','FDA','XB3','4BN','HNI','TXD','TGB','2MA','3IX','NDE','YG8','A3R','AQP','N9V','A1AU','T5A',
'TH3','A1IY','A1LT','A1BC','30N','3IR','8ID','J5H','MRS','9SO','2CP','CND','HEB','COS','XB6','S0N','48F','0WD','HFD','ZZC',
'FVH','ZNA','F8G','UKL','A1D8','SV9','V9G','ZNH','SCD','B44','P5F','Y9K','NDP','HMG','A1IG','1GZ','AR6','ZID','MF6','A1H8',
'CO6','C7G','F2R','FYN','1DG','08T','4KW','9XR','P1H','CS8','HEJ','A1A7','112','A1AH','WZG','1CZ','HIR','4TA','DG1','3ZZ',
'ATP','COW','SAE','0RQ','H5L','NMX','SDX','GRA','48N','AFH','DND','CA5','NPW','25L','A1AT','FRE','FAS','MCA','TGC','SCA',
'DAK','AV2','ONA','NO7','ST9','MFK','A1IJ','NWS','SND','NDA','A1L1','NH9','CV0','6AT','TAP','CA6','Q0U','8Z2','WYV','MLC',
'TW3','FA9','A1BO','IRC','5JB','G5P','I7I','FB0','UBG','A3D','CMC','FA8','KXM','LRP','SCO','Y65','BYC','YE1','DCC','TG8',
'2QR','CO7','ATR','ZXW','UCC','1CV','3HC','BV8','AP0','OH6','AYU','ADJ','FAO','NJP','BA3','T1G','0ET','3KK','N7Y','V0V',
'MC4','9G1','AGS','DN4','JBT','TC6','NTE','RFC','ETB','CQM','875','8HB','4CA','6CO','AVW','NHD','8I8','AVU','CA3',
'4KX','FZN','2NE','OA9','XQD','X0F','D51','0CN','1HA','PAP','HSC','M2A','8OD','OJC','OIG','OI6','12F','ZKK','FCG','NBP',
'1QD','01A','VOV','89R','AC8','FAY','NHQ','SFC','UJG','UCA','5F9','FSH','FAQ','YXR','BW9','JSQ','ZJ3','8I9','3Y0','YNC',
'6QA','YOY','SFD','76R','12D','YZS','F2N','RMW','3CD','KD9','1HE','NBC','HGG','Q0L','FAA','XHT','CAA','KMQ','HXC','0A',
'UQ3','AT4','139','4CO','WUF','XNP','93P','A2U','G3A','LRM','SO5','BVT','N6E','ZKE','RFL','A1H2','NAI','AP5','0FQ',
'A1AA','FCX','PUA','MYA','YAO','1C4','ZOD','A8P','Y0Z','9JJ','NZQ','X5T','NAE','FFI','BJW','6FA','CPN','HIF','COK','CGK',
'IVC','YDD','A1BI','YAS','WRK','HEA','50T','NHO','ODP','XF6','OAD','VER','NAJ','ZEM','7DT','D69','A99','RVK','A2N',
'CLN','GTA','A2D','B6P','5NG','62F','YAF','UOQ','ROJ','A22',]
Kyte_Doolittle={}
k=["ILE","PHE","VAL","LEU","TRP",
   "MET","ALA","GLY","CYS","TYR",
   "PRO","THR","SER","HIS","GLU",
   "ASN","GLN","ASP","LYS","ARG",
   "PCA","SEP","TPO","M0H","AYA",
   "CSX","ORN","MLY","DM0","CSO",
   "MLZ","MSE",'XCN',"OCS",'CSS',
  "XCN","XPL","PYL","AYA",
  "CSS","CXM","YCM","TOX","T8L",
   "SME","SCY","PTR","M3L","KCX",
   "JLP","CME","HYP","CGU","HIC",
    "GYS","FME","DV9","CYG","CR2",
    "CAS","BFD","ALO","7O5",'2RA',
   'DAB','CRO','DDE','DAH','MK8',
   'NLE','ALY','IAR','PHD','PBF',
   'MEQ','SMC','R0K','SAR','TRW',
   'CSD','TRQ']


v=[4.5, 2.8, 4.2, 3.8, -0.9, 
   1.9, 1.8, -0.4, 2.5, -1.3, 
   -1.6,-0.7, -0.8, -3.2, -3.5, 
   -3.5,-3.5,-3.5, -3.9,-4.5,
   -3.5,-0.8,-0.7, 2.5, 1.8,
   2.5, 1.8, -3.9, -3.9,2.5,
   -3.9,1.9,2.5, 2.5, 2.5, 
   2.5, -3.9, -3.9, 1.8,
   2.5,1.9,2.5,-0.9,-0.7,
   1.9,2.5,-1.3,-3.9,-3.9,
   -3.9,2.5,-1.6,-3.5,-3.2,
    -0.8,1.9,-3.5,2.5,-0.4,
    2.5,-3.5,-0.7,1.8, 1.8,
   1.8,-0.7, -3.2, 2.8, 3.8,
   3.8, -3.9,-4.5,-3.5, 2.8,
   -3.5,2.5, -3.5, -0.4, -0.9,
   2.5, -0.9]

for i in range(len(v)):
    Kyte_Doolittle[k[i]]=v[i]

NON_POLYMER=["UMA",'FDA', '2CL', 'ZIR', 'PO4', 'TCC', 'GFB', 'N22', 'FAD', 'GDU', 'I84', 'HEM',
 '1SM', 'Q24', 'N2P', 'H2S', 'O8A', 'JPN', '5WB', 'GOL', 'D64', '0SZ', '22Y', '0T0', 'DMS',
 'M49', 'QUE', 'D16', 'LPA', '51P', 'IUR', '79X', 'FLR', 'XCF', 'IDH', '641', 'NA0', 'UFP',
 'AOX', 'C2F', 'GUN', 'CO8', '15M', '0U6', 'STR', 'HMG', 'IOD', 'PG5', 'M42', '25U', 'VRO',
 'P22', 'ACE', 'NO3', 'FOL', 'P80', 'GCG', 'U5P', 'KPC', 'CP7', 'OHP', 'UGA', 'ZHK', 'BRF',
 '3TZ', 'MAN', 'MRO', 'EMO', '1TJ', '04J', '298', '14Q', '6ME', '1CO', 'CLZ', '0HV', 'RH1',
 'MN3', 'ONH', 'HG', '8PS', 'T1C', 'AX3', 'L34', 'X1H', 'YTR', 'VJJ', '744', 'NI', 'EEB',
 'H11', 'PYD', 'TAB', 'PG2', '0SL', 'STL', '3G4', 'N8E', 'URC', 'TRK', 'WRO', '665', 'BPY',
 'GRA', 'ITU', 'HST', 'YRO', 'CC2', 'CO4', 'MES', 'PMD', 'PS9', 'NAG', 'ERD', 'AFI', '1UQ',
 'GPB', 'DTD', 'MYC', 'CHJ', 'SAS', 'PXM', 'D2D', 'CUE', 'PGE', 'M23', '2MC', 'TXD', 'CDP',
 'OAG', 'PE5', '695', 'MRY', 'FFA', 'HHF', 'AES', '2PE', 'CBW', '62P', 'E04', 'TLO', 'OAA',
 'CI2', 'FIT', 'F42', 'YF3', 'URA', 'VAP', 'HBI', 'PYR', 'TQ6', 'EST', '393', 'SE5', 'KA5',
 'TIR', '4PI', 'TQ3', '1PQ', 'Q19', 'MXX', '1PE', 'BEL', 'UD6', 'A2Z', 'BMA', 'QLR', 'EWQ',
 '1J1', 'TCU', 'TGC', '1CY', 'CHO', 'FMN', 'QRO', 'FXY', 'DX2', 'ADP', 'Q12', 'NAI', 'IM3',
 'GDX', 'YF4', '1CS', 'WRA', 'GLC', 'P25', 'FT1', 'TNL', 'P23', '2HC', '8PC', 'VG9', 'WHF',
 'ALR', 'IMN', 'NAD', '53N', 'JPJ', 'IXF', 'CIT', '566', 'NH4', 'BRU', 'TMA', 'ANB', '3NA',
 '0VJ', 'COA', 'TN2', 'IU5', 'SPJ', 'DX1', 'SN2', 'HOH', '33T', '3T4', 'D2F', 'FLF', 'SAG',
 'SO4', 'ACT', 'TMQ', '1MM', 'S3H', 'OCS', 'FOM', 'AX6', 'Q27', 'FMT', 'DX3', 'C50', 'UFM',
 'FT3', 'EDO', 'TQ5', 'GDD', '2CY', 'FLP', 'P65', 'TRR', 'LYA', 'ORO', 'JU2', 'P1Z', 'TCT',
 'TPP', 'C18', 'LDA', 'UQ5', 'MEV', 'RZW', 'ROU', 'ACY', 'OMN', 'C0R', 'EU', 'MRE', 'VVV',
 '4HB', 'EMF', 'SPD', 'MQU', 'D2H', 'DX6', 'MPD', '3RO', 'LIH', 'UXH', 'NAP', 'TQ4', 'CB1',
 'DDQ', 'O73', 'ASE', 'RUT', '3CZ', 'ASD', '2AM', 'ZRO', 'ZOM', '7PC', 'B3P', 'JMS', 'SIN',
 'COQ', 'IMD', 'XRA', '511', 'NTI', 'SS8', 'GW3', 'IXE', 'H4B', 'PM0', 'GDP', 'FT2', 'LAC',
 '4X4', 'FES', 'HXC', '1TB', 'ARH', 'ADN', 'IPA', 'Q22', 'TOP', 'MAH', '388', '5PP', 'IZP',
 '1HR', 'COG', 'NPX', 'QSO', 'CIY', '53S', 'AKM', 'K', 'CBD', '2X9', 'ADQ', 'NJ8', 'HP7',
 '15P', 'IBH', 'UDP', 'HD1', 'THT', '16J', 'LII', 'UPP', 'UD1', 'MXA', '5OP', 'PRD', 'L24',
 'EMU', 'TRS', 'SBI', 'TUD', 'PPY', 'MN', '372', 'D09', '1W6', 'CP6', 'MIY', 'SP7', 'APR',
 '4HF', 'COM', 'BT9', 'BOG', '4HC', '173', 'QLZ', 'MDE', 'GEN', '3TU', 'OPD', 'PG4', 'CL',
 '572', 'BFI', 'AX8', 'GSH', 'A0D', 'CHV', 'BCT', 'PDC', '1UE', '5Y5', 'Q11', '352', '973',
 'RAL', 'NA', 'SEC','022', 'FRU', 'Q74', 'SR', '07M', '6DR', 'ACJ', 'TN5', '201', 'MEF',
 'DHP', 'ROD', '1L5', 'CAA', '09T', 'DM2', 'LIT', 'OXY', 'PEG', 'RAR', 'JPA', 'NFL', 'C2U',
 'TH3', 'AB3', 'XRO', 'NDP', 'IXG', '1CV', 'EOH', 'FM6', 'IBP', 'TLT', 'DHF', 'ZES', 'RJ1',
 'EQI', 'COP', 'MG', 'TQT', '238', 'DVP', 'DUR', 'MMV', 'ETE', 'J01', '55V', 'JP1', 'Q20',
 'REN', 'SSB', 'JPM', 'ZN', 'TON', 'L37', 'A7B', 'HD2', 'SO3', 'COO', 'CPS', 'NN1', 'DDD',
 'AMP', '1R0', '7DR', 'ID5', 'DH1', 'I7T', '8PG', 'E09', 'CB3', 'T3', 'AU', 'AX5', 'RJ6',
 'TES', '5DR', 'P33', '4I5', 'SB', 'TCL', 'LMR', 'D2E', 'TGG', 'ZPG', '53T', '0U5', 'XMP',
 'DOY', 'LMT', 'A2P', '19V', 'FX4', 'I2H', 'ITB', 'CIE', 'PDN', 'ZZ0', 'NBC', 'ZXZ', 'ITA',
 'YTZ', 'VGD', 'NPS', '465', '53V', 'UMP', 'DCQ', 'HTK', 'URM', 'WRB', 'DTT', '47D', 'P6G',
 'D3E', 'BR', '14M', '21V', 'EKB', 'VMY', 'UPG', 'TOL', 'BIK', 'A26', 'LDP', 'J4Z', 'D2B',
 '468', 'AND', 'FT0', 'XYL', 'EPU', 'HXS', 'D2R', 'GEQ', 'ZST', 'T30', '53R', '7CK', 'GHW',
 'HTG', 'NA7', 'FNR', 'VAK', 'CBO', 'DQ1', 'ROE', '3II', 'SUZ', 'MTX', 'ZMG', 'IIG', '1QZ',
 'EPE', 'D1D', 'SEC', 'BRE', '06W', 'A21', 'J5Z', 'BME', 'MCV', 'Q26', 'NSP', '245', 'MQ1',
 'AKY', '9DR', 'SF4', 'TLA', 'ACN', 'FE1', 'ID8', 'N01', 'CA', 'WUB', 'AHE', 'AYD', 'UFG',
 'PDO', 'TUI', 'VGP', 'MOT', 'SFY', 'AX1', 'HGZ', 'TDM', 'KMP', 'LDT', 'GHC', 'AOM', 'GAL',
 '06U', '5BU', 'ZEA', '340', '3LD', 'UDX', '311', '2M3', 'NCO', 'BIO', 'K2C', 'COS', 'TDK',
 '10H', '3OQ', 'DH3', 'UD2', 'DMF', 'FID''A54', 'ACO', 'ACP', 'ANP', 'AR6', 'ARR', 'ATP',
 'AXW', 'B12', 'B2S', 'BBM', 'BCO', 'C5P', 'CCL', 'CDM', 'CF9', 'CMP', 'CO', 'CO3','GTG',
 'CS',  'CTN', 'DA', 'DC', 'DCP', 'DDT', 'DG', 'DGT', 'DME', 'DND', 'DOI','PPI','TBU',
 'DP1', 'DP2', 'DP9', 'DT', 'DTP', 'DTU', 'EUI', 'F6P', 'F89', 'FBP', 'FEG', 'FYN', 'G',
 'GMP', 'GNP', 'GSU', 'GTP', 'H4W', 'H5W', 'HC5', 'HDA', 'HFG', 'HSX', 'HW0', 'HW1', 'HW4',
 'HW5', 'HW6', 'HW7', 'HW8', 'HW9', 'IMO', 'IPT', 'IZG', 'J11', 'J14', 'JI2', 'JI3', 'JI4',
 'JI5', 'JI7', 'JK4', 'JK5', 'JRR', 'JRS', 'JSR', 'JSS', 'K2Q', 'LAB', 'LAR', 'LY3', 'LYB',
 'LYD', 'M0H', 'MCA', 'MEK', 'MKK', 'MLC', 'MPO', 'MRA', 'MTL', 'NDS', 'NLG', 'NNR', 'NOS',
 'OPE', 'ORN', 'P0D', 'PAL', 'PCX', 'PD8', 'PFG', 'PGS', 'PLP', 'POP', 'PPS', 'PRP', 'PRX',
 'PT3', 'PXL', 'PXT', 'Q05', 'Q10', 'Q13', 'Q14', 'Q15', 'QJ2', 'QJ4', 'QJ7', 'QJ8',
 'RBF', 'RPD', 'RPL', 'SAM', 'SCA', 'SCH', 'SCN', 'SFD', 'SMI', 'TAC', 'TAD', 'TAM', 'TFA',
 'THF', 'THG', 'THM', 'TIZ', 'TLL', 'TMF', 'TPS', 'TTP', 'TZP', 'UNK', 'V04', 'VDO',
 'VIR', 'VLB', 'VRA', 'X2C', 'X2D', 'X2E', 'XFJ', 'XFK', 'XFL', 'XFM', 'Y2B', 'YGP',
 'YQY', 'ZBA','01A', '08R', '147', '1CX', '1ET', '1EV', '1EW', '1Q6', '1Q7', '2P7',
 '332', '3OR', '3OS', '3PO', '3V0', '3XA', '3XC', '3XD', '3XF', '4BM', '4HX', '59R',
 '59W', '5AD', '5GP', '5Y0', '77D', '7MC', '8AX', '8BX', '8CX', 'A54', 'FID','0N6','0TK','0U9', '10M', '12H', '12J','1PG','13P', '141', '14V', '16G', '198', '1N7',  '1YN',
   '243', '24B', '250', '25W', '290', '2CH', '2K8',  '2OW',  '31K','XYP',
 '37X', '3L1', '3S4', '3S5',  '3XV', '3Y8', '40E', '4CH',  '4HA',  '4IP', '4KE', '5N1', '5VD',
  'CP',  'CRT',  'CTE', 'CTP', 'CU', 'CUA', 'CXO', 'CYN', 'D10', 'DAA',
 'DGG', 'DGP', 'DHE', 'DHK','6E9', '6MO', '6PC', '6XR', '75O', '7DP', '7KZ',
 '7OX', '7WF', '8AC', '8H6', '8LG', '8RB', '92N', '9CS', '9D7', '9ON', '9ZZ', 'A1S', 'A2G', 'AC1',
 'ADE', 'ADX', 'AER', 'AF3', 'AGA', 'AGS', 'AKG', 'AKR', 'ALF',  'AMZ',
 'ARS', 'ASC', 'AZI', 'BA', 'BCB', 'BCL', 'BEF', 'BEN', 'BGC', 'BGL',
 'BHG', 'BLM', 'BNG', 'BPB', 'BPH', 'BTB', 'C0G', 'CAC', 'CD', 'CE1',
 'CHD', 'CMO', 'CMS', 'CNV', 'CNX', 'CO2', 'COD', 'DIO', 'DLZ', 'DMN', 'DNC', 'DP6', 'DPO', 'DTB',
 'DWM','DXC', 'DXP', 'DYY', 'E1T', 'E23', 'ECN', 'EP1', 'EPH', 'EUG', 'F21', 'F3S',
  'F4S', 'FAH', 'FC6', 'FCA', 'FCO', 'FDP', 'FE', 'FE2', 'FLA','FUM',
 'FLC', 'FMX', 'FOA', 'FOR', 'FUC', 'FUH', 'DTL', 'DTN', 'DTV', 'DTZ',
 'G1T', 'GAI', 'GAP', 'GBL', 'GD', 'GKD', 'GKE', 'GLA', 'GLV', 'GOA', 'GP1', 'GP7',
 'HAS', 'HBX', 'HC6',  'HEC', 'HEO', 'HEZ',  'HG1', 'HGS', 'HHR',
 'HIU', 'HQO', 'HY0', 'HZN', 'I3P',  'I4P','IHP', 'ILO', 'INS', 'IPH', 'IT9', 'JB2', 'K2B', 'KEN',
 'KOU', 'LBP', 'LFA', 'LFR', 'LI', 'LIP', 'LOP', 'M59', 'MCN', 'MD1',
 'MEU', 'MGD', 'MGF', 'MGT', 'MHF', 'MJZ', 'ML1', 'MLA', 'MLI',
 'MLT', 'MO', 'MOS', 'MOW', 'MQ7', 'MQ8', 'MRD', 'MTA', 'MTE', 'MXN',
 'MYR', 'MYS', 'MZR', 'NBA', 'NCA', 'NEN', 'NEP', 'NFU', 'NG1',
 'NGA', 'NH2', 'NH3','NHE', 'NIR', 'NMY', 'NO', 'NPI', 'NRF',
 'NYP', 'O', 'OCA', 'OCT', 'ODT', 'OLC', 'OMO', 'OSM','4FO','DTH',
 'OXL', 'OXM', 'OXT', 'OZ2', 'P2S', 'P3A', 'P4G', 'PAP','YCP','4HP',
 'PAU', 'PC', 'PE3', 'PE4', 'PEF', 'PEO', 'PEP', 'PER', 'PEU', 'PG0', 'PG6',
 'PHB',  'PHS', 'PL3', 'PLM', 'PM7', 'POB', 'PPQ', 'PPV', 'PQQ', 'PR', 'PRF',
 'PTY', 'PXO', 'RB', 'RIB', 'RNH', 'RNP', 'RUB', 'S3P', 'SAH',  'SBO', 'SH2',
 'SIA', 'SIS', 'SM', 'SMA', 'SNP', 'SP5', 'SRO', 'ST9', 'STE', 'STU', 'T3A',
 'T3Q', 'TA1', 'TAR', 'TBR',  'TCE', 'TCF', 'TDR', 'TEA', 'TEC', 'TEO',
 'TFB' 'TG1', 'THJ','UQ1', 'UUL','UVW', 'VO4', 'W', 'WDT', 'WJ1', 'XAX',
 'TOE', 'TOY', 'TRT', 'TRV',  'TSR', 'TXP', 'TYD', 'UAG', 'UNL', 'UNX','CGN',
 'XE', 'XPO', 'XUL',  'Y7Y', 'Y83',  'YLG', 'ZZ1','2OP','HF2','6F5','HCI','BTN','PDM']
# This  polar hydrogen's names correspond to that of the program Reduce. 
polarHydrogens = {}
polarHydrogens["ALA"] = ["H"]
polarHydrogens["GLY"] = ["H"]
polarHydrogens["SER"] = ["H", "HG"]
polarHydrogens["THR"] = ["H", "HG1"]
polarHydrogens["LEU"] = ["H"]
polarHydrogens["ILE"] = ["H"]
polarHydrogens["VAL"] = ["H"]
polarHydrogens["ASN"] = ["H", "HD21", "HD22"]
polarHydrogens["GLN"] = ["H", "HE21", "HE22"]
polarHydrogens["ARG"] = ["H", "HH11", "HH12", "HH21", "HH22", "HE"]
polarHydrogens["HIS"] = ["H", "HD1", "HE2"]
polarHydrogens["TRP"] = ["H", "HE1"]
polarHydrogens["PHE"] = ["H"]
polarHydrogens["TYR"] = ["H", "HH"]
polarHydrogens["GLU"] = ["H"]
polarHydrogens["ASP"] = ["H"]
polarHydrogens["LYS"] = ["H", "HZ1", "HZ2", "HZ3"]
polarHydrogens["PRO"] = []
polarHydrogens["CYS"] = ["H"]
polarHydrogens["MET"] = ["H"]
polarHydrogens["PCA"] = []


  # Dictionary from an H atom to its donor atom.
donorAtom = {}
donorAtom["H"] = "N"
# ARG
# ARG NHX
# Angle: NH1, HH1X, point and NH2, HH2X, point 180 degrees.
# ARG NE
# Angle: ~ 120 NE, HE, point, 180 degrees
donorAtom["HH11"] = "NH1"
donorAtom["HH12"] = "NH1"
donorAtom["HH21"] = "NH2"
donorAtom["HH22"] = "NH2"
donorAtom["HE"] = "NE"
 # ASN
 # Angle ND2,HD2X: 180
 # Plane: CG,ND2,OD1
 # Angle CG-OD1-X: 120
donorAtom["HD21"] = "ND2"
donorAtom["HD22"] = "ND2"
  # GLU
  # PLANE: CD-OE1-OE2
  # ANGLE: CD-OEX: 120
  # GLN
  # PLANE: CD-OE1-NE2
  # Angle NE2,HE2X: 180
  # ANGLE: CD-OE1: 120
donorAtom["HE21"] = "NE2"
donorAtom["HE22"] = "NE2"

  # HIS Donors: ND1, NE2
  # Angle ND1-HD1 : 180
  # Angle NE2-HE2 : 180
donorAtom["HD1"] = "ND1"
donorAtom["HE2"] = "NE2"

  # TRP Donor: NE1-HE1
  # Angle NE1-HE1 : 180
donorAtom["HE1"] = "NE1"

  # LYS Donor NZ-HZX
  # Angle NZ-HZX : 180
donorAtom["HZ1"] = "NZ"
donorAtom["HZ2"] = "NZ"
donorAtom["HZ3"] = "NZ"

  # TYR donor: OH-HH
  # Angle: OH-HH 180
donorAtom["HH"] = "OH"
  # SER donor:
  # Angle: OG-HG-X: 180
donorAtom["HG"] = "OG"

  # THR donor:
  # Angle: OG1-HG1-X: 180
donorAtom["HG1"] = "OG1"

    # Dictionary from acceptor atom to a third atom on which to compute the plane.
acceptorPlaneAtom = {}
acceptorPlaneAtom["O"] = "CA"
  # ASN Acceptor
acceptorPlaneAtom["OD1"] = "CB"
  # ASP
  # Plane: CB-CG-OD1
  # Angle CG-ODX-point: 120
acceptorPlaneAtom["OD2"] = "CB"
  # GLU
  # PLANE: CD-OE1-OE2
  # ANGLE: CD-OEX: 120
  # GLN
  # PLANE: CD-OE1-NE2
  # Angle NE2,HE2X: 180
  # ANGLE: CD-OE1: 120
acceptorPlaneAtom["OE1"] = "CG"
acceptorPlaneAtom["OE2"] = "CG"
  # HIS Acceptors: ND1, NE2
  # Plane ND1-CE1-NE2
  # Angle: ND1-CE1 : 125.5
  # Angle: NE2-CE1 : 125.5
acceptorPlaneAtom["ND1"] = "NE2"
acceptorPlaneAtom["NE2"] = "ND1"
  # TYR acceptor OH
  # Plane: CE1-CZ-OH
  # Angle: CZ-OH 120
acceptorPlaneAtom["OH"] = "CE1"

  # TYR donor: OH-HH
  # Angle: OH-HH 180
acceptorPlaneAtom["OH"] = "CE1"


    # Dictionary from an acceptor atom to its directly bonded atom on which to
  # compute the angle.
acceptorAngleAtom = {}
acceptorAngleAtom["O"] = "C"
acceptorAngleAtom["O1"] = "C"
acceptorAngleAtom["O2"] = "C"
acceptorAngleAtom["OXT"] = "C"
acceptorAngleAtom["OT1"] = "C"
acceptorAngleAtom["OT2"] = "C"
  # ASN Acceptor
acceptorAngleAtom["OD1"] = "CG"
  # Angle CG-ODX-point: 120
acceptorAngleAtom["OD2"] = "CG"

  # GLU
  # PLANE: CD-OE1-OE2
  # ANGLE: CD-OEX: 120
  # GLN
  # PLANE: CD-OE1-NE2
  # Angle NE2,HE2X: 180
  # ANGLE: CD-OE1: 120

acceptorAngleAtom["OE1"] = "CD"
acceptorAngleAtom["OE2"] = "CD"

  # HIS Acceptors: ND1, NE2
  # Plane ND1-CE1-NE2
  # Angle: ND1-CE1 : 125.5
  # Angle: NE2-CE1 : 125.5
acceptorAngleAtom["ND1"] = "CE1"
acceptorAngleAtom["NE2"] = "CE1"
  # TYR acceptor OH
  # Plane: CE1-CZ-OH
  # Angle: CZ-OH 120
acceptorAngleAtom["OH"] = "CZ"
  # SER acceptor:
  # Angle CB-OG-X: 120
acceptorAngleAtom["OG"] = "CB"
  # THR acceptor:
  # Angle: CB-OG1-X: 120
acceptorAngleAtom["OG1"] = "CB"
# this is the van der walls distances
distVDW=dict()
a=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
b=[170,120,152,155,180,190,180,175,198,227,275,173,139,231,194,100]
for each in range(16):
    distVDW[a[each]]=b[each]/100

