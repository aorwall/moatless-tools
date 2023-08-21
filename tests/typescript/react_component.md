# Chunk 1
```tsx
import React, { useState } from 'react';

// Define a type for the props
interface MyComponentProps {
  initialCount?: number;
}

// Define the component
// ...
```

# Chunk 2
```tsx


// ...
const MyComponent: React.FC<MyComponentProps> = ({ initialCount = 0 }) => {
  // Use a state variable
  const [count, setCount] = useState(initialCount);

  const resetCount = () => {
    setCount(initialCount);
  };

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
}

export default MyComponent;

```

