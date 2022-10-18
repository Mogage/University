package containterFactory;

import container.Container;
import utils.Constants.Strategy;

public interface Factory {
    Container createContainer(Strategy strategy);
}
