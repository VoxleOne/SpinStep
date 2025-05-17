# Troubleshooting & FAQ

## Why is my traversal slow?
- Check the size of your DiscreteOrientationSet. Very fine grids may result in millions of steps.
- Try reducing the number of points or increasing `angle_threshold`.
- For large sets, consider spatial pruning or batching.

## I got a ValueError about quaternions?
- All quaternions must be normalized and non-zero. Use the provided utilities or `scipy.spatial.transform.Rotation` to generate valid quaternions.

## What if my node doesn't have `.orientation` or `.children`?
- All nodes must implement both attributes. If you use custom node types, subclass or adapt as needed.

## Can I use Euler angles or axis-angle?
- Yes! Convert to quaternions using `scipy.spatial.transform.Rotation`.

## What is a "closure property" test?
- For symmetry groups, applying all group elements in sequence should bring you (numerically) back to a group element—verifying group structure.

---
