from cidistance import main

# Ab: I, VII, V, I, VII, V, I, Eb: VII, Ab: V, I
main.get_shortest_paths([(8,1,1),(8,1,7),(8,1,5),(8,1,1),(8,1,7),(8,1,5),(8,1,1),(3,1,7),(8,1,5),(8,1,1)]) # argument is the list of (tonic, mode, degree)

# Ab: I, VII, V, I, VII, V, I, Eb: VII, I, Ab: V, I
main.get_shortest_paths([(8,1,1),(8,1,7),(8,1,5),(8,1,1),(8,1,7),(8,1,5),(8,1,1),(3,1,7),(3,1,1),(8,1,5),(8,1,1)])
