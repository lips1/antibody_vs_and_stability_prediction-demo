#!/usr/bin/env python
"""Quick test of viscosity predictor"""

from predictor import predictor

# Test sequences
test_sequences = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVKLSDKDIFSQHIQVQQDQPDFTAPVHYVKVNVKQDPPHHPAPGTXVP",
    "MVSTAASGLS",
    "MKTAILSLYIFSFLFVN"
]

print("=" * 60)
print("Testing Viscosity Predictor with Real Sequences")
print("=" * 60)

for i, seq in enumerate(test_sequences, 1):
    try:
        print(f"\n[Test {i}] Sequence: {seq[:50]}...")
        result = predictor.predict(seq)
        print(f"  ✓ Viscosity: {result:.3f} cP")
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")

print("\n" + "=" * 60)
print("Test complete!")
