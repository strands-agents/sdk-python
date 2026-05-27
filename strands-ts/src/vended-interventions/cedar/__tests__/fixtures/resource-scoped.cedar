// Allow deleting only records you own
permit(
  principal,
  action == Action::"delete",
  resource is Record
) when {
  resource.owner == principal
};

// Allow reading any record
permit(
  principal,
  action == Action::"read",
  resource is Record
);
