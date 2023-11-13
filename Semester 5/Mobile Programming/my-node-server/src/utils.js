export const jwtConfig = {
  secret:
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
};

export const exceptionHandler = async (ctx, next) => {
  try {
    return await next();
  } catch (err) {
    console.log(err);
    ctx.body = { message: err.message || "Unexpected error." };
    ctx.status = err.status || 500;
  }
};

export const timingLogger = async (ctx, next) => {
  const start = Date.now();
  await next();
  console.log(
    `${ctx.method} ${ctx.url} => ${ctx.response.status}, ${
      Date.now() - start
    }ms`
  );
};
