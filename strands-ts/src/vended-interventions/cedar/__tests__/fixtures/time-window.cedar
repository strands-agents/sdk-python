// Allow all tools during business hours (9-17 UTC)
permit(
  principal,
  action,
  resource
) when {
  context.session.hour_utc >= 9 && context.session.hour_utc < 17
};
