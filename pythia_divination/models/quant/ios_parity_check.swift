// On-device parity check for the SeekingBeta.AI iOS port of the 48-feature options LSTM.
//
// Verifies that the Swift feature engine + options extractor reproduce the server's
// pythia feature_engine / options_features numerically. Compile WITH the two iOS source
// files (this file must be named main.swift for swiftc top-level code), against the golden
// fixture emitted by `convert_to_tflite.py golden`:
//
//   python models/quant/convert_to_tflite.py golden          # writes /tmp/sb_tflite/golden_parity.json
//   cp models/quant/ios_parity_check.swift /tmp/main.swift
//   swiftc -O <ios>/Core/OptionsFeatureExtractor.swift <ios>/Core/QuantFeatureEngine.swift /tmp/main.swift -o /tmp/sb_parity
//   /tmp/sb_parity                                            # expect "PARITY OK"
//
// Last run: FEATURES max abs diff 1.86e-09 ; OPTIONS exact (0.0).
import Foundation

struct StockPriceBar { let date: Date; let open, high, low, close, volume: Double }

struct Golden: Decodable {
    struct Bar: Decodable { let epoch: Double; let open, high, low, close, volume: Double }
    struct ChainRow: Decodable { let strike, iv, bid, volume, oi: Double; let isCall: Bool; let exp_epoch: Double }
    let feature_columns: [String]
    let expected_features_last: [Double]
    let bars: [Bar]
    let spot: Double
    let asof_epoch: Double
    let chain: [ChainRow]
    let expected_options: [String: Double]
}

let data = try! Data(contentsOf: URL(fileURLWithPath: "/tmp/sb_tflite/golden_parity.json"))
let g = try! JSONDecoder().decode(Golden.self, from: data)
guard g.feature_columns == QuantFeatureEngine.featureColumns else { print("FEATURE ORDER MISMATCH"); exit(1) }

let bars = g.bars.map {
    StockPriceBar(date: Date(timeIntervalSince1970: $0.epoch),
                  open: $0.open, high: $0.high, low: $0.low, close: $0.close, volume: $0.volume)
}
let last = QuantFeatureEngine().featureRows(bars: bars, options: nil, fearIndex: 20.0).last!
var maxFeatDiff = 0.0
var worstFeat = ""
for (i, name) in QuantFeatureEngine.featureColumns.enumerated() {
    let diff = abs(last[i] - g.expected_features_last[i])
    if diff > maxFeatDiff { maxFeatDiff = diff; worstFeat = name }
}
print(String(format: "FEATURES: max abs diff = %.3e  (worst: %@)", maxFeatDiff, worstFeat))

let contracts = g.chain.map {
    OptionContract(strike: $0.strike, impliedVol: $0.iv, bid: $0.bid, volume: $0.volume,
                   openInterest: $0.oi, isCall: $0.isCall, expiration: Date(timeIntervalSince1970: $0.exp_epoch))
}
let opt = OptionsFeatureExtractor().extract(contracts: contracts, spot: g.spot,
                                            asOf: Date(timeIntervalSince1970: g.asof_epoch))!
var maxOptDiff = 0.0
for k in OptionsFeatureExtractor.featureColumns {
    let diff = abs((opt[k] ?? .nan) - (g.expected_options[k] ?? .nan))
    if diff > maxOptDiff { maxOptDiff = diff }
}
print(String(format: "OPTIONS: max abs diff = %.3e", maxOptDiff))
let ok = maxFeatDiff < 1e-3 && maxOptDiff < 1e-3
print(ok ? "PARITY OK" : "PARITY FAIL")
exit(ok ? 0 : 1)
