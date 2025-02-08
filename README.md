# The Ray Tracer Challenge in F#

This repository is a Rust implementation of the ray tracer found in [*The Ray Tracer Challenge: A Test-Driven Guide to Your First 3D Renderer*](https://pragprog.com/titles/jbtracer/the-ray-tracer-challenge/) by Jamis Buck.

The Rust implementation here began as an exercise to learn Rust by porting my existing [F# implementation of this ray tracer](https://github.com/bmitc/the-ray-tracer-challenge-fsharp) to Rust. The idea was to dive into an already implemented and designed project in a not so foreign language to Rust in order to focus on how to do basic things in Rust like represent types, design traits, implement functions on types, implement traits, write tests, etc. while also learning practical syntax and semantics through this exercise.

While the general approach has been to port over the F# code, I have endeavored, to the best of my current knowledge of Rust, to be idiomatic to Rust. Part of this approach, though, has been to treat Rust almost as if it is a garbage-collected language and to favor immutability over mutability still. Since this is my first Rust project within just a couple weeks of learning Rust, there are undoubtedly several areas that need improvement, and the hope is that I can refactor and fix things as I learn more and more Rust.
