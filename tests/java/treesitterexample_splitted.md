# Chunk 1
```java
package com.treesitter.example;

// ...

public class TreeSitterExample {

    // ...

    public TreeSitterExample() {
        myVariable = 20;
    }

    // ...
}
```

# Chunk 2
```java
package com.treesitter.example;

// ...

public class TreeSitterExample {

    // ...

    public void myMethod(int parameter) {
        myVariable = parameter;

        if (parameter > 5) {
            System.out.println("Parameter is greater than 5");
        }

        for (int i = 0; i < 10; i++) {
            System.out.println("For loop iteration: " + i);
        }

        // ...
    }

    // ...
}
```

# Chunk 3
```java
package com.treesitter.example;

// ...

public class TreeSitterExample {

    // ...

    public void myMethod(int parameter) {

        // ...

        while (parameter < 10) {
            System.out.println("While loop, parameter: " + parameter);
            parameter++;
        }

        do {
            System.out.println("Do-while loop, parameter: " + parameter);
            parameter--;
        } while (parameter > 0);

        // ...
    }

    // ...
}
```

# Chunk 4
```java
package com.treesitter.example;

// ...

public class TreeSitterExample {

    // ...

    public void myMethod(int parameter) {

        // ...

        switch (parameter) {
            case 1:
                System.out.println("Parameter is 1");
                break;
            default:
                System.out.println("Parameter is default");
                break;
        }

        // ...
    }

    // ...
}
```

# Chunk 5
```java
package com.treesitter.example;

// ...

public class TreeSitterExample {

    // ...

    public void myMethod(int parameter) {

        // ...

        try {
            Random random = new Random();
            if (random.nextBoolean()) {
                throw new Exception("Random exception");
            }
        } catch (Exception e) {
            System.out.println("Caught exception: " + e.getMessage());
        } finally {
            System.out.println("Finally block executed");
        }
    }

    // ...
}
```

# Chunk 6
```java
package com.treesitter.example;

// ...

public class TreeSitterExample {

    // ...

    interface MathOperation {
        int operation(int a, int b);
    }

    // ...
}
```

# Chunk 7
```java
package com.treesitter.example;

// ...

public class TreeSitterExample {

    // ...

    public enum Colors {
        RED, GREEN, BLUE;
    }

    // ...
}
```

# Chunk 8
```java
package com.treesitter.example;

// ...

public class TreeSitterExample {

    // ...

    public @interface MyAnnotation {
        String value() default "";
    }

    // ...
}
```

# Chunk 9
```java
package com.treesitter.example;

// ...

public class TreeSitterExample {

    // ...

    public interface MyInterface {
        void myInterfaceMethod();
    }

    // ...
}
```

# Chunk 10
```java
package com.treesitter.example;

// ...

public class TreeSitterExample {

    // ...

    public static void main(String[] args) {
        TreeSitterExample example = new TreeSitterExample();
        example.myMethod(3);
    }
}
```

# Chunk 11
```java
package com.treesitter.example;

// ...

public class TreeSitterExample {

    int myVariable = 10;

    // ...

    MathOperation addition = (a, b) -> {
        return a + b;
    };

    // ...
}
```

# Chunk 12
```java
package com.treesitter.example;

import java.util.Random;

// ...
```

