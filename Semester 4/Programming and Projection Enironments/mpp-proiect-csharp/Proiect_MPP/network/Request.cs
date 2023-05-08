using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace network
{
    public enum RequestType
    {
        LOGIN, LOGOUT, BUY_TICKET, GET_FLIGHT, GET_DD_FLIGHT, GET_AIRPORT, GET_ALL_AIRPORTS, GET_A_FLIGHTS
    }

    [Serializable]
    public class Request
    {
        private Request() { }

        public RequestType Type { get; private set; }

        public object Data { get; private set; }

        public override string ToString()
        {
            return "Request{" +
                    "type='" + Type + '\'' +
                    ", data='" + Data + '\'' +
                    '}';
        }

        public class Builder
        {
            private readonly Request request = new Request();

            public Builder Type(RequestType type)
            {
                request.Type = type;
                return this;
            }

            public Builder Data(Object data)
            {
                request.Data = data;
                return this;
            }

            public Request Build()
            {
                return request;
            }
        }
    }
}
