Decision

Why does os.walk() return [] instead of None for empty directories?

Reason:

Returning an empty list allows developers to iterate directly without checking for None, reducing conditional logic and making code simpler and safer.

Decision

Why does os.walk() return filenames instead of full paths?

Reason:

The root path is already provided separately. Storing only filenames avoids repeating the same directory path for every file, making the API more memory-efficient. Developers can reconstruct the full path when needed using os.path.join(root, filename).